#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data

from inference import (
    _move_reaction_dict_to_device,
    encode_proteins,
    load_model_from_ckpt,
)


def _natural_sort_key(s: str) -> Tuple:
    """Sort '2' before '10' when IDs are numeric strings."""
    import re

    parts = re.split(r"(\d+)", str(s))
    key: List = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p)
    return tuple(key)


@torch.no_grad()
def encode_all_reactions(
    model,
    reaction_dict: Dict,
    reaction_ids: List[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Returns float tensor [R, H] on CPU."""
    rows: List[torch.Tensor] = []
    bs = max(1, int(batch_size))
    for start in range(0, len(reaction_ids), bs):
        batch_ids = reaction_ids[start : start + bs]
        rxn_data_list: List[Data] = []
        for rid in batch_ids:
            rxn = reaction_dict[rid]
            rxn_data_list.append(_move_reaction_dict_to_device(rxn, device))
        out = model.encode_reactions(rxn_data_list)  # [B, H]
        rows.append(out.detach().float().cpu())
    return torch.cat(rows, dim=0)


@torch.no_grad()
def encode_all_proteins(
    model,
    protein_dict: Dict[str, torch.Tensor],
    protein_ids: List[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Returns float tensor [P, H] on CPU; encodes in chunks to limit GPU memory."""
    bs = max(1, int(batch_size))
    chunks: List[torch.Tensor] = []
    for start in range(0, len(protein_ids), bs):
        sub_ids = protein_ids[start : start + bs]
        chunks.append(
            encode_proteins(model, protein_dict, sub_ids, device).detach().float().cpu()
        )
    return torch.cat(chunks, dim=0)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(script_dir, "extended_test", "WJIN")

    parser = argparse.ArgumentParser(
        description="Similarity matrix CSV: rows=protein_id, columns=reaction_id."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=default_dir,
        help=f"Folder containing reaction.pt and protein.pt (default: {default_dir})",
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Trained model .ckpt path")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path (default: <data_dir>/similarity_matrix.csv)",
    )
    parser.add_argument("--reaction_pt", type=str, default=None, help="Override reaction.pt path")
    parser.add_argument("--protein_pt", type=str, default=None, help="Override protein.pt path")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument(
        "--side",
        type=str,
        default="both",
        choices=["substrate", "product", "both"],
        help="Override reaction encoding side",
    )
    parser.add_argument(
        "--reaction_batch_size",
        type=int,
        default=8,
        help="Reactions per forward when encoding",
    )
    parser.add_argument(
        "--protein_batch_size",
        type=int,
        default=256,
        help="Proteins per forward when encoding (GPU memory)",
    )
    parser.add_argument(
        "--sort_keys",
        type=str,
        default="natural",
        choices=["natural", "lexical"],
        help="Order of proteins (rows) and reactions (columns): natural or lexical",
    )
    parser.add_argument(
        "--matrix_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Where to compute R @ P.T (cpu avoids large GPU alloc; cuda can be faster)",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    reaction_pt = args.reaction_pt or os.path.join(data_dir, "reaction_whole.pt")
    protein_pt = args.protein_pt or os.path.join(data_dir, "protein.pt")
    out_csv = args.output_csv or os.path.join(data_dir, "similarity_matrix_whole.csv")

    if not os.path.isfile(reaction_pt):
        raise FileNotFoundError(f"reaction_pt not found: {reaction_pt}")
    if not os.path.isfile(protein_pt):
        raise FileNotFoundError(f"protein_pt not found: {protein_pt}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    reaction_dict: Dict = torch.load(reaction_pt)
    protein_dict: Dict[str, torch.Tensor] = torch.load(protein_pt)

    key_fn = _natural_sort_key if args.sort_keys == "natural" else (lambda x: str(x))
    reaction_ids = sorted(reaction_dict.keys(), key=key_fn)
    protein_ids = sorted(protein_dict.keys(), key=key_fn)

    R, P = len(reaction_ids), len(protein_ids)
    if R == 0 or P == 0:
        raise RuntimeError(f"Empty reaction or protein set: R={R}, P={P}")

    print(f"Loading model from {args.ckpt}")
    model = load_model_from_ckpt(args.ckpt, device=device, side_override=args.side)

    print(f"Encoding {R} reactions (batch_size={args.reaction_batch_size})...")
    reaction_repr = encode_all_reactions(
        model, reaction_dict, reaction_ids, device, args.reaction_batch_size
    )
    print(f"Encoding {P} proteins (batch_size={args.protein_batch_size})...")
    protein_repr = encode_all_proteins(
        model, protein_dict, protein_ids, device, args.protein_batch_size
    )

    mat_dev = torch.device(args.matrix_device if args.matrix_device == "cuda" else "cpu")
    if mat_dev.type == "cuda" and not torch.cuda.is_available():
        mat_dev = torch.device("cpu")
    rr = reaction_repr.to(mat_dev)
    pr = protein_repr.to(mat_dev)
    scores = rr @ pr.t()  # [R, P]
    scores_cpu = scores.float().cpu().numpy()

    rxn_cols = [str(k) for k in reaction_ids]
    prot_rows = [str(k) for k in protein_ids]
    # scores_cpu[r, p] = sim(reaction r, protein p) -> CSV: rows=protein, columns=reaction
    df = pd.DataFrame(scores_cpu.T, index=prot_rows, columns=rxn_cols)
    df.index.name = "protein_id"
    out_dir = os.path.dirname(os.path.abspath(out_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_csv, float_format="%.8g")
    print(f"Saved similarity matrix [{P} proteins x {R} reactions] to {out_csv}")


if __name__ == "__main__":
    main()
