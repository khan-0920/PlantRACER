#!/usr/bin/env python3
"""
  python find_reaction_centers.py

"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem


def parse_rxn_smiles(rxn: str) -> Tuple[Chem.Mol, Chem.Mol]:
    parts = rxn.strip().split(">>")
    if len(parts) != 2:
        raise ValueError(f"Reaction SMILES with single '>>' expected, got {len(parts)-1} arrows")
    left = Chem.MolFromSmiles(parts[0])
    right = Chem.MolFromSmiles(parts[1])
    if left is None or right is None:
        raise ValueError("RDKit cannot parse the SMILES of one side of the reaction")
    return left, right


def _max_map(mol: Chem.Mol) -> int:
    return max((a.GetAtomMapNum() for a in mol.GetAtoms()), default=0)


def assign_missing_atom_maps(
    r_mol: Chem.Mol, p_mol: Chem.Mol
) -> Tuple[Set[int], int]:
    """
    Assign new numbers to atoms with map==0. Return (new assigned map number set, next available number).
    """
    nxt = max(_max_map(r_mol), _max_map(p_mol)) + 1
    new_maps: Set[int] = set()
    for mol in (r_mol, p_mol):
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                atom.SetAtomMapNum(nxt)
                new_maps.add(nxt)
                nxt += 1
    return new_maps, nxt


def _maps_in_mol(mol: Chem.Mol) -> Set[int]:
    return {a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomMapNum() > 0}


def _atom_by_map(mol: Chem.Mol, m: int) -> Optional[Chem.Atom]:
    for a in mol.GetAtoms():
        if a.GetAtomMapNum() == m:
            return a
    return None


def _neighbor_signature(mol: Chem.Mol, m: int) -> Tuple[Tuple[int, int], ...]:
    """Each neighbor is (adjacent atom map, bond type enum value), used to distinguish only bond level change (e.g. alcohol → ketone)."""
    a = _atom_by_map(mol, m)
    if a is None:
        return tuple()
    pairs = []
    for nb in a.GetNeighbors():
        nm = nb.GetAtomMapNum()
        if nm <= 0:
            continue
        b = mol.GetBondBetweenAtoms(a.GetIdx(), nb.GetIdx())
        if b is None:
            continue
        pairs.append((nm, int(b.GetBondType())))
    return tuple(sorted(pairs))


def find_centers_and_shell(
    r_mol: Chem.Mol, p_mol: Chem.Mol, new_maps: Set[int]
) -> Tuple[Set[int], Set[int]]:
    """
    Return (First: reaction center map set, Second: atoms directly connected to First and not in First).
    """
    r_maps = _maps_in_mol(r_mol)
    p_maps = _maps_in_mol(p_mol)
    common = r_maps & p_maps

    base_changed_mapped: Set[int] = set()
    for m in common:
        if _neighbor_signature(r_mol, m) != _neighbor_signature(p_mol, m):
            base_changed_mapped.add(m)

    # 仅一侧出现的「原始」map（断裂/新成键片段）；新赋 map 只靠与 base_changed 邻接规则加入
    only_r_orig = (r_maps - p_maps) - new_maps
    only_p_orig = (p_maps - r_maps) - new_maps

    first: Set[int] = set(base_changed_mapped) | only_r_orig | only_p_orig

    for m in new_maps:
        for mol in (r_mol, p_mol):
            a = _atom_by_map(mol, m)
            if a is None:
                continue
            for nb in a.GetNeighbors():
                if nb.GetAtomMapNum() in base_changed_mapped:
                    first.add(m)
                    break

    second: Set[int] = set()
    for m in first:
        for mol in (r_mol, p_mol):
            a = _atom_by_map(mol, m)
            if a is None:
                continue
            for nb in a.GetNeighbors():
                nm = nb.GetAtomMapNum()
                if nm > 0 and nm not in first:
                    second.add(nm)

    return first, second


def _format_maps(s: Set[int]) -> str:
    return ";".join(str(x) for x in sorted(s))


def process_row(rxn_mapped: str) -> Tuple[str, str, str]:
    """
    Return (rxn_with_all_maps, First, Second); if parsing fails, raise an exception.
    """
    r0 = Chem.MolFromSmiles(rxn_mapped.split(">>")[0])
    p0 = Chem.MolFromSmiles(rxn_mapped.split(">>")[1])
    if r0 is None or p0 is None:
        raise ValueError("Unsolvable mapped rxn")
    r_mol = Chem.Mol(r0)
    p_mol = Chem.Mol(p0)
    new_maps, _ = assign_missing_atom_maps(r_mol, p_mol)
    left = Chem.MolToSmiles(r_mol)
    right = Chem.MolToSmiles(p_mol)
    rxn_full = f"{left}>>{right}"
    first, second = find_centers_and_shell(r_mol, p_mol, new_maps)
    return rxn_full, _format_maps(first), _format_maps(second)


def main() -> int:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Determine reaction centers and adjacent atoms from mapped reactions")
    ap.add_argument(
        "-i",
        "--input",
        type=Path,
        default=root / "WJIN reaction_mapped.csv",
        help="CSV containing the rxn (mapped) column",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=root / "WJIN reaction_centers.csv",
        help="Output CSV",
    )
    ap.add_argument(
        "--rxn-column",
        default="rxn",
        help="Column name containing the mapped reactions (default: rxn)",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1

    with args.input.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("CSV has no headers", file=sys.stderr)
            return 1
        rows: List[Dict[str, str]] = list(reader)
        base_fields = list(reader.fieldnames)

    if args.rxn_column not in base_fields:
        print(f"Column {args.rxn_column} not found", file=sys.stderr)
        return 1

    out_rows: List[Dict[str, str]] = []
    for i, row in enumerate(rows):
        rxn = row.get(args.rxn_column, "").strip()
        new_row = dict(row)
        if not rxn:
            new_row["rxn_all_mapped"] = ""
            new_row["First"] = ""
            new_row["Second"] = ""
            out_rows.append(new_row)
            continue
        try:
            full, first, second = process_row(rxn)
            new_row["rxn_all_mapped"] = full
            new_row["First"] = first
            new_row["Second"] = second
        except Exception as e:
            print(f"Row {i+2} processing failed: {e}", file=sys.stderr)
            new_row["rxn_all_mapped"] = rxn
            new_row["First"] = ""
            new_row["Second"] = ""
        out_rows.append(new_row)

    # Ensure column order: original columns + three new columns (deduplicated)
    tail = [c for c in ["rxn_all_mapped", "First", "Second"] if c not in base_fields]
    final_fields = base_fields + tail

    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=final_fields, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in final_fields})

    print(f"Written {len(out_rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
