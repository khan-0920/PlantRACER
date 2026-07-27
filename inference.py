#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlantRACER inference / evaluation.

Evaluate a trained checkpoint on test_pair_filtered.csv:
1) Build reaction -> [enzymes] positive dictionary from pair CSV
2) Encode reactions with the CLIP reaction encoder
3) Encode proteins with the CLIP protein encoder
4) For each reaction, rank similarity against all candidate proteins
5) Compute topK hit rate and enrichment factor (EF@K)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Set, Tuple, Optional

import torch
import pandas as pd
from torch_geometric.data import Data

from model import EnzymeReactionCLIPModel
from process_dataset import atom_featurizer, bond_featurizer


def _parse_topk(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return [1, 5, 10, 20]
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return sorted(set(out))


def _strip_state_dict_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Lightning checkpoints usually store keys like "model.xxx.*"
    # Try stripping common prefixes.
    prefixes = ("model.", "model.model.", "module.model.", "module.")
    new_sd: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
                break
        new_sd[nk] = v
    return new_sd


def _infer_model_hparams_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    """
    Returns: (hidden_dim, protein_emb_dim, mpnn_layers)
    """
    hidden_dim = None
    protein_emb_dim = None

    # protein_encoder.mlp.net[0] is Linear(in_dim -> hidden_dim)
    target_suffixes = [
        "protein_encoder.mlp.net.0.weight",
        "protein_encoder.mlp.net.0.bias",
    ]

    for k, v in state_dict.items():
        if k.endswith("protein_encoder.mlp.net.0.weight"):
            # weight shape: [hidden_dim, protein_emb_dim]
            hidden_dim = int(v.shape[0])
            protein_emb_dim = int(v.shape[1])
            break

    if hidden_dim is None or protein_emb_dim is None:
        raise RuntimeError("Failed to infer hidden_dim / protein_emb_dim from checkpoint state_dict")

    # mpnn_layers: count indices in message_mlps / update_mlps
    mpnn_layers_set: Set[int] = set()
    for k in state_dict.keys():
        # Example: substrate_encoder.message_mlps.0.0.weight
        if "substrate_encoder.message_mlps." in k and ".0.weight" in k:
            try:
                midx = k.split("substrate_encoder.message_mlps.")[1].split(".")[0]
                mpnn_layers_set.add(int(midx))
            except Exception:
                pass

    if not mpnn_layers_set:
        # fallback: try update_mlps
        for k in state_dict.keys():
            if "substrate_encoder.update_mlps." in k and ".0.weight" in k:
                try:
                    midx = k.split("substrate_encoder.update_mlps.")[1].split(".")[0]
                    mpnn_layers_set.add(int(midx))
                except Exception:
                    pass

    if not mpnn_layers_set:
        raise RuntimeError("Failed to infer mpnn_layers from checkpoint state_dict")

    mpnn_layers = max(mpnn_layers_set) + 1
    return hidden_dim, protein_emb_dim, mpnn_layers


def load_model_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    side_override: Optional[str] = None,
) -> EnzymeReactionCLIPModel:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    if not isinstance(state_dict, dict):
        raise RuntimeError("Invalid checkpoint format: missing state_dict")

    state_dict = _strip_state_dict_prefix(state_dict)

    hidden_dim, protein_emb_dim, mpnn_layers = _infer_model_hparams_from_state_dict(state_dict)

    # side is an attribute only affecting encode logic. Try to read it from hyper_parameters if present.
    side: Optional[str] = None
    hp = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}
    if isinstance(hp, dict) and "side" in hp:
        side = str(hp["side"]).strip()
    if side_override:
        side = side_override
    if side not in ("substrate", "product", "both"):
        side = "substrate"

    node_dim = atom_featurizer.dim + 3  # reaction-center one-hot dims
    edge_dim = bond_featurizer.dim

    model = EnzymeReactionCLIPModel(
        node_dim=node_dim,
        edge_dim=edge_dim,
        protein_emb_dim=protein_emb_dim,
        hidden_dim=hidden_dim,
        mpnn_layers=mpnn_layers,
        dropout=0.1,  # dropout doesn't affect weights loading; keep default
        side=side,  # used as default in encode_reactions
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warning] Missing keys: {len(missing)} (show first 5): {missing[:5]}")
    if unexpected:
        print(f"[warning] Unexpected keys: {len(unexpected)} (show first 5): {unexpected[:5]}")

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def encode_proteins(
    model: EnzymeReactionCLIPModel,
    protein_dict: Dict[str, torch.Tensor],
    protein_ids: List[str],
    device: torch.device,
) -> torch.Tensor:
    emb_list: List[torch.Tensor] = []
    for uid in protein_ids:
        emb = protein_dict.get(uid)
        if emb is None:
            continue
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb, dtype=torch.float32)
        emb_list.append(emb.float().to(device))

    if not emb_list:
        raise RuntimeError("No protein embeddings found for candidate protein_ids")

    protein_emb_batch = torch.stack(emb_list, dim=0)  # [P, Dp]
    protein_repr = model.encode_proteins(protein_emb_batch)  # [P, H]
    return protein_repr


def _move_reaction_dict_to_device(rxn: Dict, device: torch.device) -> Data:
    # reaction dict contains lists of PyG Data
    subs = [g.to(device) for g in (rxn.get("substrates") or [])]
    prods = [g.to(device) for g in (rxn.get("products") or [])]
    return Data(substrates=subs, products=prods)


def main():
    parser = argparse.ArgumentParser(description="PlantRACER inference evaluator (CLIP matching)")
    parser.add_argument("--ckpt", type=str, required=True, help="Lightning checkpoint .ckpt path")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing reaction.pt / protein.pt / test_pair_filtered.csv")
    parser.add_argument("--test_pair_csv", type=str, default=None, help="test_pair_filtered.csv path (default: data_dir/test_pair_filtered.csv)")
    parser.add_argument("--reaction_pt", type=str, default=None, help="reaction.pt path (default: data_dir/reaction.pt)")
    parser.add_argument("--protein_pt", type=str, default=None, help="protein.pt path (default: data_dir/protein.pt)")
    parser.add_argument("--topk", type=str, default="1,5,10,20", help="Comma-separated K list")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--reaction_batch_size", type=int, default=8, help="How many reactions to encode per batch")
    parser.add_argument("--side", type=str, default=None, choices=["substrate", "product", "both"], help="Override model side for encoding")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(script_dir, "data_extended")
    test_pair_csv = args.test_pair_csv or os.path.join(data_dir, "test_pair_filtered.csv")
    reaction_pt_path = args.reaction_pt or os.path.join(data_dir, "reaction.pt")
    protein_pt_path = args.protein_pt or os.path.join(data_dir, "protein.pt")

    if not os.path.isfile(test_pair_csv):
        raise FileNotFoundError(f"test_pair_csv not found: {test_pair_csv}")
    if not os.path.isfile(reaction_pt_path):
        raise FileNotFoundError(f"reaction_pt not found: {reaction_pt_path}")
    if not os.path.isfile(protein_pt_path):
        raise FileNotFoundError(f"protein_pt not found: {protein_pt_path}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    topk_list = _parse_topk(args.topk)
    max_k = max(topk_list) if topk_list else 1

    print(f"Loading reaction/protein tensors from: {reaction_pt_path}, {protein_pt_path}")
    # Keep it consistent with train.py: direct torch.load without map_location.
    reaction_dict: Dict[int, Dict] = torch.load(reaction_pt_path)
    protein_dict: Dict[str, torch.Tensor] = torch.load(protein_pt_path)

    df = pd.read_csv(test_pair_csv)
    if "RHEA_ID" not in df.columns or "UniprotID" not in df.columns:
        raise ValueError("test_pair_filtered.csv must contain columns: RHEA_ID, UniprotID")

    # (1) reaction -> list of positive enzymes
    reaction_to_pos: Dict[int, Set[str]] = {}
    positive_pairs: Set[Tuple[int, str]] = set()
    all_protein_ids: Set[str] = set()
    reactions_in_csv: Set[int] = set()

    for _, row in df.iterrows():
        try:
            rhea_id = str(row["RHEA_ID"])
            uid = str(row["UniprotID"]).strip()
        except Exception:
            continue
        if not uid:
            continue
        all_protein_ids.add(uid)
        reactions_in_csv.add(rhea_id)
        positive_pairs.add((rhea_id, uid))
        reaction_to_pos.setdefault(rhea_id, set()).add(uid)

    # Filter candidates by availability in protein.pt / reaction.pt
    protein_ids = [uid for uid in sorted(all_protein_ids) if uid in protein_dict.keys()]
    if not protein_ids:
        raise RuntimeError("No candidate proteins found in protein.pt")

    reaction_ids = [rid for rid in sorted(reactions_in_csv) if rid in reaction_dict.keys()]
    if not reaction_ids:
        raise RuntimeError("No reactions found in reaction.pt")

    print(f"Candidates: reactions={len(reaction_ids)}, proteins={len(protein_ids)}, positive_pairs={len(positive_pairs)}")

    # Ensure requested K doesn't exceed candidate protein count
    topk_list = [k for k in topk_list if k <= len(protein_ids)]
    if not topk_list:
        raise RuntimeError(f"All requested topk are > number of protein candidates (P={len(protein_ids)})")
    max_k = max(topk_list)

    # (2) load model from ckpt
    model = load_model_from_ckpt(args.ckpt, device=device, side_override=args.side)

    # (2) protein embeddings -> protein representations (once)
    protein_repr = encode_proteins(model, protein_dict, protein_ids, device)  # [P, H]
    # NOTE: protein_repr already normalized in model.encode_proteins

    P = len(protein_ids)
    R = len(reaction_ids)
    total_positives = len(positive_pairs)
    if total_positives == 0:
        raise RuntimeError("No positive pairs found in test_pair_filtered.csv")

    # accumulators
    hits_any: Dict[int, int] = {k: 0 for k in topk_list}  # number of reactions with >=1 positive in topk
    pos_in_topk_total: Dict[int, int] = {k: 0 for k in topk_list}  # total #positive proteins within topk across reactions

    def process_reaction_batch(batch_rids: List[int]):
        # Encode a batch of reactions -> [B,H]
        rxn_data_list: List[Data] = []
        for rid in batch_rids:
            rxn = reaction_dict[rid]
            rxn_data = _move_reaction_dict_to_device(rxn, device)
            rxn_data_list.append(rxn_data)
        reaction_repr = model.encode_reactions(rxn_data_list)  # [B,H]

        # similarity scores against all proteins
        # reaction_repr: [B,H], protein_repr: [P,H]
        scores = reaction_repr @ protein_repr.t()  # [B,P]
        topk_scores, topk_idx = torch.topk(scores, k=max_k, dim=1, largest=True, sorted=True)

        for i, rid in enumerate(batch_rids):
            predicted_uids = [protein_ids[j] for j in topk_idx[i].tolist()]
            pos_uids = reaction_to_pos.get(rid, set())

            for k in topk_list:
                preds_k = set(predicted_uids[:k])
                inter = preds_k.intersection(pos_uids)
                if len(inter) > 0:
                    hits_any[k] += 1
                pos_in_topk_total[k] += len(inter)

    bs = max(1, int(args.reaction_batch_size))
    for start in range(0, len(reaction_ids), bs):
        end = min(len(reaction_ids), start + bs)
        process_reaction_batch(reaction_ids[start:end])

    # (5) enrichment factor
    # EF@K = (observed positives fraction in topK) / (random baseline fraction)
    # observed fraction = pos_in_topk_total[k] / (R*K)
    # random baseline fraction = total_positives / (R*P)
    print("\n===== Evaluation =====")
    print(f"Reactions: {R}  Proteins: {P}  Positive pairs: {total_positives}")
    for k in topk_list:
        ef_k = (pos_in_topk_total[k] / (R * k)) / (total_positives / (R * P))
        hit_k = hits_any[k] / R
        recall_like = pos_in_topk_total[k] / (R * k)
        print(
            f"@{k}: EF={ef_k:.6f}  HitRate={hit_k:.6f}  PosFractionInTopK={recall_like:.6f}"
        )


if __name__ == "__main__":
    main()

