#!/usr/bin/env python3
"""
Quick AID Alignment Checker
---------------------------

Checks alignment between:
- Validation ground-truth AIDs
- index_to_aid.json from FAISS index
- aid_map.pkl content keys

Prints coverage stats and a few examples of mismatches to diagnose issues early.
"""

import json
import pickle
from pathlib import Path
import sys
import os

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.utils.aid_utils import canonicalize_aid_ascii


def main():
    val_path = config.VAL_SPLIT_JSON_PATH
    index_map_path = config.INDEX_TO_AID_PATH
    aid_map_path = config.AID_MAP_PATH

    if not val_path.exists():
        print(f"[ERROR] Validation split not found: {val_path}")
        return 1

    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    gt = set()
    for item in val_data:
        for aid in item.get("relevant_aids", []):
            gt.add(canonicalize_aid_ascii(aid))

    if not index_map_path.exists():
        print(f"[WARN] index_to_aid.json not found: {index_map_path}")
        index_aids = set()
    else:
        with open(index_map_path, "r", encoding="utf-8") as f:
            index_aids = set(canonicalize_aid_ascii(a) for a in json.load(f))

    if not aid_map_path.exists():
        print(f"[WARN] aid_map.pkl not found: {aid_map_path}")
        aid_keys = set()
    else:
        with open(aid_map_path, "rb") as f:
            loaded = pickle.load(f)
        try:
            aid_keys = set(canonicalize_aid_ascii(k) for k in loaded.keys())
        except Exception:
            aid_keys = set(loaded.keys())

    def pct(num, den):
        return (num / den * 100.0) if den else 0.0

    in_index = len([a for a in gt if a in index_aids])
    in_map = len([a for a in gt if a in aid_keys])

    print("=== AID ALIGNMENT CHECK ===")
    print(f"Validation items: {len(val_data)}")
    print(f"Unique GT AIDs: {len(gt)}")
    print(f"Index AIDs: {len(index_aids)} | Coverage: {in_index}/{len(gt)} ({pct(in_index, len(gt)):.2f}%)")
    print(f"AID map keys: {len(aid_keys)} | Coverage: {in_map}/{len(gt)} ({pct(in_map, len(gt)):.2f}%)")

    missing_index = [a for a in gt if a not in index_aids][:20]
    missing_map = [a for a in gt if a not in aid_keys][:20]
    if missing_index:
        print("\nMissing in index (first 20):")
        for a in missing_index:
            print("  ", a)
    if missing_map:
        print("\nMissing in aid_map (first 20):")
        for a in missing_map:
            print("  ", a)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


