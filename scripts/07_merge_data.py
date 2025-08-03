#!/usr/bin/env python3
"""
Merge du lieu training cho Bi-Encoder va Cross-Encoder
"""

import sys
import os
import json
from pathlib import Path

# Them thu muc goc vao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
import logging
from core import setup_logging

setup_logging()


def merge_bi_encoder_data():
    """Merge du lieu Bi-Encoder tu easy va hard negatives"""
    logging.info("[BUILD] Merge du lieu Bi-Encoder...")

    # Load easy triplets
    easy_triplets = []
    if config.TRAIN_TRIPLETS_EASY_PATH.exists():
        with open(config.TRAIN_TRIPLETS_EASY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    easy_triplets.append(json.loads(line.strip()))

    # Load hard triplets
    hard_triplets = []
    if config.TRAIN_TRIPLETS_HARD_NEG_PATH.exists():
        with open(config.TRAIN_TRIPLETS_HARD_NEG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    hard_triplets.append(json.loads(line.strip()))

    # Merge triplets
    merged_triplets = easy_triplets + hard_triplets

    # Luu merged triplets
    with open(config.BI_ENCODER_TRAIN_MIXED_PATH, "w", encoding="utf-8") as f:
        for triplet in merged_triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + "\n")

    logging.info("[OK] Bi-Encoder data merging completed successfully!")
    logging.info(f"[FILE] Output saved to: {config.BI_ENCODER_TRAIN_MIXED_PATH}")
    logging.info(f"[INFO] Easy triplets: {len(easy_triplets)}")
    logging.info(f"[INFO] Hard triplets: {len(hard_triplets)}")
    logging.info(f"[INFO] Total triplets: {len(merged_triplets)}")

    return True


def merge_cross_encoder_data():
    """Merge du lieu Cross-Encoder tu easy va hard negatives"""
    logging.info("[BUILD] Merge du lieu Cross-Encoder...")

    # Load easy pairs
    easy_pairs = []
    if config.TRAIN_PAIRS_PATH.exists():
        with open(config.TRAIN_PAIRS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    easy_pairs.append(json.loads(line.strip()))

    # Load hard pairs
    hard_pairs = []
    if config.TRAIN_PAIRS_HARD_NEG_PATH.exists():
        with open(config.TRAIN_PAIRS_HARD_NEG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    hard_pairs.append(json.loads(line.strip()))

    # Merge pairs
    merged_pairs = easy_pairs + hard_pairs

    # Luu merged pairs
    with open(config.TRAIN_PAIRS_MIXED_PATH, "w", encoding="utf-8") as f:
        for pair in merged_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logging.info("[OK] Cross-Encoder data merging completed successfully!")
    logging.info(f"[FILE] Output saved to: {config.TRAIN_PAIRS_MIXED_PATH}")
    logging.info(f"[INFO] Easy pairs: {len(easy_pairs)}")
    logging.info(f"[INFO] Hard pairs: {len(hard_pairs)}")
    logging.info(f"[INFO] Total pairs: {len(merged_pairs)}")

    return True


def main():
    """Ham chinh"""
    logging.info("[START] Bat dau merge du lieu...")

    try:
        # Merge Bi-Encoder data
        if not merge_bi_encoder_data():
            logging.error("[FAIL] Loi khi merge Bi-Encoder data")
            return False

        # Merge Cross-Encoder data
        if not merge_cross_encoder_data():
            logging.error("[FAIL] Loi khi merge Cross-Encoder data")
            return False

        logging.info("[SUCCESS] Tat ca du lieu da duoc merge thanh cong!")
        return True

    except Exception as e:
        logging.error(f"[FAIL] Loi khi merge du lieu: {e}")
        return False


if __name__ == "__main__":
    main()
