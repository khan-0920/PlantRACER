#!/usr/bin/env python3
"""
Use RXNmapper to generate atom map numbers for each reaction SMILES in the CSV.

Dependencies (must be installed): rxnmapper, rdkit, torch, transformers

Usage:
  python map_reactions_rxnmapper.py
  python map_reactions_rxnmapper.py -i "WJIN reaction.csv" -o "WJIN reaction_mapped.csv"
  python map_reactions_rxnmapper.py --batch-size 16
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

from rxnmapper import RXNMapper


def _chunked(xs: List[str], size: int) -> List[List[str]]:
    return [xs[i : i + size] for i in range(0, len(xs), size)]


def map_reactions(
    mapper: RXNMapper,
    rxns: List[str],
    batch_size: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for batch in _chunked(rxns, batch_size):
        out.extend(mapper.get_attention_guided_atom_maps(batch))
    return out


def main() -> int:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="RXNmapper batch atom mapping")
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=root / "WJIN reaction.csv",
        help="Input CSV (must contain rxn column)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=root / "WJIN reaction_mapped.csv",
        help="Output CSV",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of reactions per batch sent to the model (reduce if out of memory)",
    )
    args = p.parse_args()

    if not args.input.is_file():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1

    with args.input.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "rxn" not in reader.fieldnames:
            print("CSV must contain headers and contain rxn column", file=sys.stderr)
            return 1
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    rxns = [r["rxn"].strip() for r in rows]
    mapper = RXNMapper()
    try:
        mapped_list = map_reactions(mapper, rxns, args.batch_size)
    except Exception as e:
        print(f"RXNmapper failed: {e}", file=sys.stderr)
        return 1

    others = [c for c in fieldnames if c != "rxn"]
    out_fields = others + ["rxn_original", "rxn", "mapping_confidence"]

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for row, item in zip(rows, mapped_list):
            orig = row["rxn"]
            mapped = item.get("mapped_rxn") or orig
            conf = item.get("confidence", "")
            new_row = {c: row[c] for c in others}
            new_row["rxn_original"] = orig
            new_row["rxn"] = mapped
            new_row["mapping_confidence"] = conf
            writer.writerow(new_row)

    print(f"Written {len(rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
