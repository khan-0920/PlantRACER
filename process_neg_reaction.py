#!/usr/bin/env python3


import os
from typing import Dict, List, Any, Set

import pandas as pd
import torch
from tqdm import tqdm

from process_dataset import (
    parse_cano_rxn,
    smiles_list_to_graphs,
    smiles_list_to_graphs_with_reaction_center,
    _parse_reaction_center_field,
)


def build_neg_reaction_dict_from_csv(
    csv_path: str,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    从 neg_reaction_with_atoms.csv 构建:
        {
            reaction_id: [
                {"substrates": [Data...], "products": [Data...]},
                ...
            ]
        }

    CSV 要求至少包含列:
        - reaction_id
        - atom_map: 反应 SMILES（带或不带 atom map 均可）
        - First: 反应中心原子的 atom map id 列表 (可选)
        - Second: 反应中心周围一圈原子的 atom map id 列表 (可选)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"neg_reaction CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"reaction_id", "atom_map"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"neg_reaction_with_atoms.csv 缺少必要列: {missing}")

    use_first = "First" in df.columns
    use_second = "Second" in df.columns

    neg_reaction_dict: Dict[int, List[Dict[str, Any]]] = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="build_neg_reaction_dict"):
        rid = str(row["reaction_id"])
        rxn_str = row.get("atom_map")
        if not isinstance(rxn_str, str) or not rxn_str.strip():
            continue
        rxn_str = rxn_str.strip()

        subs_smi, prods_smi = parse_cano_rxn(rxn_str)
        if subs_smi is None or prods_smi is None:
            continue

        center_ids: Set[int] = set()
        if use_first:
            center_ids = _parse_reaction_center_field(row.get("First"))

        second_ids: Set[int] = set()
        if use_second:
            second_ids = _parse_reaction_center_field(row.get("Second"))

        if center_ids or second_ids:
            substrates = smiles_list_to_graphs_with_reaction_center(
                subs_smi,
                reaction_center_map_ids=center_ids,
                second_shell_map_ids=second_ids,
            )
            products = smiles_list_to_graphs_with_reaction_center(
                prods_smi,
                reaction_center_map_ids=center_ids,
                second_shell_map_ids=second_ids,
            )
        else:
            substrates = smiles_list_to_graphs(subs_smi)
            products = smiles_list_to_graphs(prods_smi)

        if substrates is None or products is None:
            continue

        entry = {"substrates": substrates, "products": products}
        if rid not in neg_reaction_dict:
            neg_reaction_dict[rid] = []
        neg_reaction_dict[rid].append(entry)

    return neg_reaction_dict


def run(data_dir: str = None) -> int:
    """
    主入口:
        - 读取 data/neg_reaction.csv
        - 构建 neg_reaction_dict
        - 保存为 data/neg_reaction.pt
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = data_dir or os.path.join(script_dir, "data_extended")
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "neg_reaction_with_atoms.csv")
    out_path = os.path.join(data_dir, "neg_reaction.pt")

    neg_reaction_dict = build_neg_reaction_dict_from_csv(csv_path)
    torch.save(neg_reaction_dict, out_path)

    print(f"Done. Saved {len(neg_reaction_dict)} reaction IDs to {out_path}")
    return len(neg_reaction_dict)


if __name__ == "__main__":
    run()

