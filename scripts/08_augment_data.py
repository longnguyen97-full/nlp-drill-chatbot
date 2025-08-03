#!/usr/bin/env python3
"""
Augment du lieu training de tang cuong hieu suat mo hinh
"""

import sys
import os
import json
import random
from pathlib import Path

# Them thu muc goc vao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
import logging
from core import setup_logging

setup_logging()


def augment_triplets_data(input_path, output_path, augmentation_factor=2.0):
    """Augment du lieu triplets cho Bi-Encoder"""
    logging.info(f"[BUILD] Augment triplets tu {input_path}")

    # Load triplets
    triplets = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                triplets.append(json.loads(line.strip()))

    logging.info(f"[INFO] Loaded {len(triplets)} triplets")

    # Check if we have data to augment
    if len(triplets) == 0:
        logging.warning("[WARNING] No triplets to augment, creating empty file")
        with open(output_path, "w", encoding="utf-8") as f:
            pass
        logging.info("[OK] Triplets augmentation completed (empty)")
        logging.info(f"[FILE] Output saved to: {output_path}")
        logging.info(f"[INFO] Original triplets: 0")
        logging.info(f"[INFO] Augmented triplets: 0")
        logging.info(f"[INFO] Augmentation factor: 0.00")
        return True

    # Augment triplets
    augmented_triplets = []

    for triplet in triplets:
        # Them triplet goc
        augmented_triplets.append(triplet)

        # Tao cac bien the
        query = triplet.get("query", "")
        positive = triplet.get("positive", "")
        negative = triplet.get("negative", "")

        # Bien the 1: Them dau cau
        if random.random() < 0.3:
            augmented_triplets.append(
                {"query": query + "?", "positive": positive, "negative": negative}
            )

        # Bien the 2: Them tu khoa phap luat
        if random.random() < 0.2:
            legal_keywords = ["theo quy dinh", "can cu", "theo luat"]
            keyword = random.choice(legal_keywords)
            augmented_triplets.append(
                {
                    "query": f"{keyword} {query}",
                    "positive": positive,
                    "negative": negative,
                }
            )

        # Bien the 3: Thay doi thu tu tu
        if random.random() < 0.1:
            words = query.split()
            if len(words) > 3:
                random.shuffle(words)
                augmented_triplets.append(
                    {
                        "query": " ".join(words),
                        "positive": positive,
                        "negative": negative,
                    }
                )

    # Luu augmented triplets
    with open(output_path, "w", encoding="utf-8") as f:
        for triplet in augmented_triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + "\n")

    logging.info("[OK] Triplets augmentation completed successfully!")
    logging.info(f"[FILE] Output saved to: {output_path}")
    logging.info(f"[INFO] Original triplets: {len(triplets)}")
    logging.info(f"[INFO] Augmented triplets: {len(augmented_triplets)}")

    # Fix division by zero
    if len(triplets) > 0:
        augmentation_factor_actual = len(augmented_triplets) / len(triplets)
        logging.info(f"[INFO] Augmentation factor: {augmentation_factor_actual:.2f}")
    else:
        logging.info(f"[INFO] Augmentation factor: 0.00")

    return True


def augment_pairs_data(input_path, output_path, augmentation_factor=1.3):
    """Augment du lieu pairs cho Cross-Encoder"""
    logging.info(f"[BUILD] Augment pairs tu {input_path}")

    # Load pairs
    pairs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line.strip()))

    logging.info(f"[INFO] Loaded {len(pairs)} pairs")

    # Check if we have data to augment
    if len(pairs) == 0:
        logging.warning("[WARNING] No pairs to augment, creating empty file")
        with open(output_path, "w", encoding="utf-8") as f:
            pass
        logging.info("[OK] Pairs augmentation completed (empty)")
        logging.info(f"[FILE] Output saved to: {output_path}")
        logging.info(f"[INFO] Original pairs: 0")
        logging.info(f"[INFO] Augmented pairs: 0")
        logging.info(f"[INFO] Augmentation factor: 0.00")
        return True

    # Augment pairs
    augmented_pairs = []

    for pair in pairs:
        # Them pair goc
        augmented_pairs.append(pair)

        # Tao cac bien the
        texts = pair.get("texts", [])
        label = pair.get("label", 0)

        if len(texts) >= 2:
            query = texts[0]
            passage = texts[1]

            # Bien the 1: Them dau cau cho query
            if random.random() < 0.3:
                augmented_pairs.append(
                    {"texts": [query + "?", passage], "label": label}
                )

            # Bien the 2: Them tu khoa phap luat
            if random.random() < 0.2:
                legal_keywords = ["theo quy dinh", "can cu", "theo luat"]
                keyword = random.choice(legal_keywords)
                augmented_pairs.append(
                    {"texts": [f"{keyword} {query}", passage], "label": label}
                )

            # Bien the 3: Thay doi thu tu tu trong query
            if random.random() < 0.1:
                words = query.split()
                if len(words) > 3:
                    random.shuffle(words)
                    augmented_pairs.append(
                        {"texts": [" ".join(words), passage], "label": label}
                    )

    # Luu augmented pairs
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in augmented_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logging.info("[OK] Pairs augmentation completed successfully!")
    logging.info(f"[FILE] Output saved to: {output_path}")
    logging.info(f"[INFO] Original pairs: {len(pairs)}")
    logging.info(f"[INFO] Augmented pairs: {len(augmented_pairs)}")

    # Fix division by zero
    if len(pairs) > 0:
        augmentation_factor_actual = len(augmented_pairs) / len(pairs)
        logging.info(f"[INFO] Augmentation factor: {augmentation_factor_actual:.2f}")
    else:
        logging.info(f"[INFO] Augmentation factor: 0.00")

    return True


def augment_bi_encoder_data():
    """Augment du lieu Bi-Encoder"""
    return augment_triplets_data(
        config.BI_ENCODER_TRAIN_MIXED_PATH,
        config.BI_ENCODER_TRAIN_AUGMENTED_PATH,
        augmentation_factor=1.5,
    )


def augment_cross_encoder_data():
    """Augment du lieu Cross-Encoder"""
    return augment_pairs_data(
        config.TRAIN_PAIRS_MIXED_PATH,
        config.TRAIN_PAIRS_AUGMENTED_PATH,
        augmentation_factor=1.3,
    )


def main():
    """Ham chinh"""
    logging.info("[START] Bat dau augment du lieu...")

    try:
        # Augment Bi-Encoder data
        if not augment_bi_encoder_data():
            logging.error("[FAIL] Loi khi augment Bi-Encoder data")
            return False

        # Augment Cross-Encoder data
        if not augment_cross_encoder_data():
            logging.error("[FAIL] Loi khi augment Cross-Encoder data")
            return False

        logging.info("[SUCCESS] Tat ca du lieu da duoc augment thanh cong!")
        return True

    except Exception as e:
        logging.error(f"[FAIL] Loi khi augment du lieu: {e}")
        return False


if __name__ == "__main__":
    main()
