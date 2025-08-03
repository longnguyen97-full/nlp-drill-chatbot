#!/usr/bin/env python3
"""
Prepare Training Data - Script Chuẩn Bị Dữ Liệu Training Tối Ưu
===============================================================

Script này chuẩn bị dữ liệu training cho Bi-Encoder và Cross-Encoder
sử dụng dữ liệu đã được fix mapping và tối ưu hóa.

Tác giả: LawBot Team
Phiên bản: Optimized Training Data v2.0
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import random
import sys

sys.path.append(str(Path(__file__).parent.parent))
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_fixed_data():
    """Load dữ liệu đã fix"""
    logger.info("Loading fixed training data...")

    # Load fixed training data
    train_fixed_path = config.DATA_RAW_DIR / "train_fixed.json"
    with open(train_fixed_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # Load aid_map
    with open(config.AID_MAP_PATH, "rb") as f:
        aid_map = pickle.load(f)

    logger.info(f"Loaded {len(train_data)} fixed training samples")
    logger.info(f"Loaded {len(aid_map)} AIDs")

    return train_data, aid_map


def create_triplets(train_data, aid_map):
    """Tạo triplets cho Bi-Encoder training với hard negatives"""
    logger.info("Creating triplets for Bi-Encoder with hard negatives...")

    triplets = []
    all_aids = list(aid_map.keys())

    for sample in train_data:
        question = sample["question"]
        positive_aids = sample["relevant_aids"]

        # Tạo triplets với positive và hard negative samples
        for positive_aid in positive_aids:
            # Tìm hard negative aids (không có trong positive)
            negative_candidates = [aid for aid in all_aids if aid not in positive_aids]

            if negative_candidates:
                # Chọn multiple hard negatives cho mỗi positive
                num_negatives = min(3, len(negative_candidates))
                selected_negatives = random.sample(negative_candidates, num_negatives)

                for negative_aid in selected_negatives:
                    triplet = {
                        "query": question,
                        "positive": positive_aid,
                        "negative": negative_aid,
                    }
                    triplets.append(triplet)

    logger.info(f"Created {len(triplets)} triplets with hard negatives")
    return triplets


def create_pairs(train_data, aid_map):
    """Tạo pairs cho Cross-Encoder training với balanced data"""
    logger.info("Creating balanced pairs for Cross-Encoder...")

    positive_pairs = []
    negative_pairs = []
    all_aids = list(aid_map.keys())

    for sample in train_data:
        question = sample["question"]
        positive_aids = sample["relevant_aids"]

        # Positive pairs
        for positive_aid in positive_aids:
            positive_pair = {"texts": [question, positive_aid], "label": 1}
            positive_pairs.append(positive_pair)

        # Hard negative pairs (balanced)
        negative_candidates = [aid for aid in all_aids if aid not in positive_aids]
        num_negatives = min(
            len(positive_aids) * 2, len(negative_candidates)
        )  # 2x negatives

        if negative_candidates:
            selected_negatives = random.sample(negative_candidates, num_negatives)

            for negative_aid in selected_negatives:
                negative_pair = {"texts": [question, negative_aid], "label": 0}
                negative_pairs.append(negative_pair)

    # Combine and shuffle
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    logger.info(f"Created {len(positive_pairs)} positive pairs")
    logger.info(f"Created {len(negative_pairs)} negative pairs")
    logger.info(f"Total pairs: {len(all_pairs)}")

    return all_pairs


def create_evaluation_data(train_data, aid_map):
    """Tạo dữ liệu evaluation với format đúng"""
    logger.info("Creating evaluation data with correct format...")

    evaluation_data = []

    for sample in train_data:
        evaluation_sample = {
            "qid": sample["qid"],
            "question": sample["question"],
            "relevant_aids": sample["relevant_aids"],
            "all_aids": list(aid_map.keys()),
        }
        evaluation_data.append(evaluation_sample)

    logger.info(f"Created {len(evaluation_data)} evaluation samples")
    return evaluation_data


def create_validation_split(train_data, aid_map, split_ratio=0.15):
    """Tạo validation split từ training data"""
    logger.info(f"Creating validation split with ratio {split_ratio}...")

    # Shuffle data
    random.shuffle(train_data)
    split_idx = int(len(train_data) * (1 - split_ratio))

    train_split = train_data[:split_idx]
    val_split = train_data[split_idx:]

    # Create validation data
    validation_data = []
    for sample in val_split:
        validation_sample = {
            "qid": sample["qid"],
            "question": sample["question"],
            "relevant_aids": sample["relevant_aids"],
            "all_aids": list(aid_map.keys()),
        }
        validation_data.append(validation_sample)

    # Save splits
    with open(config.TRAIN_SPLIT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(train_split, f, ensure_ascii=False, indent=2)

    with open(config.VAL_SPLIT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(validation_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Created train split: {len(train_split)} samples")
    logger.info(f"Created validation split: {len(validation_data)} samples")

    return train_split, validation_data


def save_training_data(
    triplets, pairs, evaluation_data, train_split, validation_data, aid_map
):
    """Lưu dữ liệu training với format tối ưu"""
    logger.info("Saving optimized training data...")

    # Save triplets in JSONL format
    triplets_path = config.DATA_PROCESSED_DIR / "triplets.jsonl"
    with open(triplets_path, "w", encoding="utf-8") as f:
        for triplet in triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + "\n")

    # Save pairs in JSONL format
    pairs_path = config.DATA_PROCESSED_DIR / "pairs.jsonl"
    with open(pairs_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Save evaluation data
    evaluation_data_path = config.DATA_PROCESSED_DIR / "evaluation_data.json"
    with open(evaluation_data_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, ensure_ascii=False, indent=2)

    # Save validation data
    validation_data_path = config.DATA_PROCESSED_DIR / "validation_data.json"
    with open(validation_data_path, "w", encoding="utf-8") as f:
        json.dump(validation_data, f, ensure_ascii=False, indent=2)

    # Create comprehensive training stats
    training_stats = {
        "triplets_count": len(triplets),
        "pairs_count": len(pairs),
        "evaluation_samples": len(evaluation_data),
        "train_samples": len(train_split),
        "validation_samples": len(validation_data),
        "positive_pairs": len([p for p in pairs if p["label"] == 1]),
        "negative_pairs": len([p for p in pairs if p["label"] == 0]),
        "total_aids": len(aid_map),
        "mapping_fixed": True,
        "format": "optimized",
    }

    training_stats_path = config.DATA_PROCESSED_DIR / "training_stats.json"
    with open(training_stats_path, "w", encoding="utf-8") as f:
        json.dump(training_stats, f, ensure_ascii=False, indent=2)

    logger.info(f"Training stats: {training_stats}")
    logger.info("Optimized training data saved successfully!")


def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("PREPARE TRAINING DATA - OPTIMIZED VERSION")
    logger.info("=" * 60)

    try:
        # Load fixed data
        train_data, aid_map = load_fixed_data()

        # Create validation split
        train_split, validation_data = create_validation_split(train_data, aid_map)

        # Create triplets for Bi-Encoder
        triplets = create_triplets(train_split, aid_map)

        # Create pairs for Cross-Encoder
        pairs = create_pairs(train_split, aid_map)

        # Create evaluation data
        evaluation_data = create_evaluation_data(validation_data, aid_map)

        # Save all training data
        save_training_data(
            triplets, pairs, evaluation_data, train_split, validation_data, aid_map
        )

        logger.info("=" * 60)
        logger.info("TRAINING DATA PREPARATION COMPLETED!")
        logger.info("=" * 60)
        logger.info("Next steps:")
        logger.info("1. Train Bi-Encoder with optimized triplets")
        logger.info("2. Train Cross-Encoder with balanced pairs")
        logger.info("3. Run comprehensive evaluation")

    except Exception as e:
        logger.error(f"Error during training data preparation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
