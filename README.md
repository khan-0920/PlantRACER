# PlantRACER

CLIP-style contrastive learning for **enzyme–reaction matching**. Reactions are encoded as molecular graphs (substrates and products) with optional reaction-center features; proteins are encoded from precomputed ESM2 embeddings. Training uses symmetric InfoNCE loss with optional side-reaction negatives.

## Overview

Given enzyme–reaction pairs, the model learns a shared embedding space where catalyzing pairs are pulled together and non-catalyzing pairs are pushed apart. At inference time, for each reaction the model ranks candidate enzymes by cosine similarity.

**Architecture** (`model.py`):

- **MoleculeGraphEncoder** — MPNN message passing + attention pooling per molecule
- **ReactionEncoder** — self-attention over molecule-level representations
- **ProteinEncoder** — MLP on mean-pooled ESM2 embeddings (1280-d)
- **EnzymeReactionCLIPModel** — joint reaction/protein encoders with L2-normalized outputs

**Loss** (`loss.py`):

- Standard symmetric CLIP InfoNCE (`clip_loss`)
- Extended loss with side-reaction negatives in the protein→reaction direction (`clip_loss_with_negatives`)

## Project structure

```
PlantRACER/
├── train.py                  # PyTorch Lightning training
├── model.py                  # Model definitions
├── loss.py                   # CLIP loss functions
├── process_dataset.py        # Build reaction.pt and protein.pt from raw CSV/FASTA
├── process_neg_reaction.py   # Build neg_reaction.pt from side-reaction CSV
├── inference.py              # Evaluate checkpoint on test pairs (HitRate, EF@K)
├── Inference_external.py     # Full reaction–protein similarity matrix for external data
├── environment.yml           # Conda environment (PyTorch, PyG, RDKit, etc.)
├── scripts/
│   ├── map_reactions_rxnmapper.py   # Atom-map reactions with RXNmapper
│   ├── find_reaction_centers.py     # Label reaction centers (First / Second shells)
│   └── generate_side_react.py       # Generate side-reaction SMILES
└── data_extended/            # Example raw data (CSV, FASTA)
    ├── reaction_mapped_with_atoms.csv
    ├── neg_reaction_with_atoms.csv
    └── protein.fa
```

## Requirements

Core dependencies:

- Python 3.12
- PyTorch 2.5 + CUDA (optional)
- PyTorch Geometric (`torch-geometric`, `torch-scatter`, `torch-sparse`)
- PyTorch Lightning 2.4
- RDKit
- transformers (ESM2)
- pandas, numpy, scikit-learn

Install via the provided conda environment:

```bash
conda env create -f environment.yml
conda activate pyg
```

Additional tools for data preparation:

- **RXNmapper** — atom mapping (`pip install rxnmapper`)
- Local **ESM2** checkpoint — required by `process_dataset.py` for protein embeddings

## Data preparation

### Data and checkpoints
All dataset and pretrained checkpoints can download from [here](https://drive.google.com/file/d/1yCe5KR3DYU6L_WCuTuKCo8L7XDOh0jgC/view?usp=sharing)

### Expected files in `data_dir`

| File | Description |
|------|-------------|
| `reaction_mapped_with_atoms.csv` | Mapped reactions with optional `First` / `Second` reaction-center columns |
| `protein.fa` | Protein sequences (UniProt IDs in FASTA headers) |
| `train_pair_filtered.csv` | Training pairs: `RHEA_ID`, `UniprotID` |
| `test_pair_filtered.csv` | Test pairs: `RHEA_ID`, `UniprotID` |
| `neg_reaction_with_atoms.csv` | Side reactions per reaction ID (optional, for negative sampling) |

### Preprocessing pipeline

1. **Atom-map reactions** (if not already mapped):

```bash
python scripts/map_reactions_rxnmapper.py \
  -i "reaction.csv" \
  -o "reaction_mapped.csv"
```

2. **Find reaction centers** (optional, improves atom features):

```bash
python scripts/find_reaction_centers.py \
  -i reaction_mapped.csv \
  -o reaction_mapped_with_atoms.csv
```

3. **Generate side reactions** (optional):

```bash
python scripts/generate_side_react.py
```

4. **Build tensor caches**:

```bash
# Reactions + ESM2 protein embeddings -> reaction.pt, protein.pt
python process_dataset.py --data_dir data_extended

# Side-reaction graphs -> neg_reaction.pt
python process_neg_reaction.py
```

`process_dataset.py` writes:

- `reaction.pt` — `{RHEA_ID: {"substrates": [PyG Data...], "products": [PyG Data...]}}`
- `protein.pt` — `{UniprotID: Tensor[Dp]}` (mean-pooled ESM2, Dp=1280)

**Note:** Edit the `esm_ckpt_path` in `process_dataset.py` to point to your local ESM2 model directory before running.

Atom features include a 3-dimensional reaction-center encoding (center / neighbor / other) when `First` and `Second` columns are present, yielding `node_dim = 179`.

## Training

```bash
python train.py \
  --data_dir data_extended \
  --save_dir ckpt \
  --batch_size 32 \
  --epochs 100 \
  --lr 1e-4 \
  --hidden_dim 512 \
  --mpnn_layers 5 \
  --side both
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data/` | Directory with `reaction.pt`, `protein.pt`, pair CSVs, optional `neg_reaction.pt` |
| `--save_dir` | `ckpt` | Checkpoint output directory |
| `--batch_size` | 32 | Pairs per batch (unique RHEA_ID and UniProtID within each batch) |
| `--side` | `both` | Reaction encoding: `substrate`, `product`, or `both` |
| `--temperature` | 0.07 | InfoNCE temperature |
| `--strategy` | `auto` | Lightning strategy (e.g. `ddp` for multi-GPU) |

Checkpoints are saved as `save_dir/last.ckpt` (only the latest checkpoint is kept).

Training uses a custom batch sampler that ensures each batch has unique reactions and unique proteins, and each pair appears at most once per epoch.

## Inference

### Internal evaluation

Rank enzymes for each reaction on `test_pair_filtered.csv` and report enrichment factor (EF@K) and hit rate:

```bash
python inference.py \
  --ckpt ckpt/last.ckpt \
  --data_dir data_extended \
  --topk 1,5,10,20
```

### External similarity matrix

Encode all reactions and proteins in a folder and write a CSV matrix (rows = proteins, columns = reactions):

```bash
python Inference_external.py \
  --ckpt ckpt/last.ckpt \
  --data_dir /path/to/external_data \
  --output_csv similarity_matrix.csv
```

Defaults expect `reaction_whole.pt` and `protein.pt` in `data_dir`; override with `--reaction_pt` and `--protein_pt` if needed.

## License

Add license information here if applicable.
