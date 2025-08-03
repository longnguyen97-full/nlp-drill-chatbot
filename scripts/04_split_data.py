#!/usr/bin/env python3
"""
Chia du lieu thanh training va validation sets
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


def split_raw_json_data(
    input_path, train_output_path, val_output_path, val_size=0.1, random_state=42
):
    """Chia du lieu JSON thanh training va validation sets"""
    logging.info(f"[FILE] Dang chia du lieu tu {input_path}")

    # Load du lieu
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logging.info(f"[INFO] Tong so mau: {len(data)}")

    # Set random seed
    random.seed(random_state)

    # Shuffle du lieu
    random.shuffle(data)

    # Tinh toan split point
    val_count = int(len(data) * val_size)
    train_count = len(data) - val_count

    # Chia du lieu
    train_data = data[:train_count]
    val_data = data[val_count:]

    logging.info(f"[INFO] Training samples: {len(train_data)}")
    logging.info(f"[INFO] Validation samples: {len(val_data)}")

    # Luu training data
    with open(train_output_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    # Luu validation data
    with open(val_output_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    logging.info("[OK] Raw data splitting completed successfully!")
    logging.info(f"[FILE] Training data saved to: {train_output_path}")
    logging.info(f"[FILE] Validation data saved to: {val_output_path}")

    return True


def split_jsonl_data(
    input_path, train_output_path, val_output_path, val_size=0.1, random_state=42
):
    """Chia du lieu JSONL thanh training va validation sets"""
    logging.info(f"[FILE] Dang chia du lieu JSONL tu {input_path}")

    # Load du lieu
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))

    logging.info(f"[INFO] Tong so mau: {len(data)}")

    # Set random seed
    random.seed(random_state)

    # Shuffle du lieu
    random.shuffle(data)

    # Tinh toan split point
    val_count = int(len(data) * val_size)
    train_count = len(data) - val_count

    # Chia du lieu
    train_data = data[:train_count]
    val_data = data[val_count:]

    logging.info(f"[INFO] Training samples: {len(train_data)}")
    logging.info(f"[INFO] Validation samples: {len(val_data)}")

    # Luu training data
    with open(train_output_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Luu validation data
    with open(val_output_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logging.info("[OK] JSONL data splitting completed successfully!")
    logging.info(f"[FILE] Training data saved to: {train_output_path}")
    logging.info(f"[FILE] Validation data saved to: {val_output_path}")

    return True


def split_raw_training_data():
    """Chia du lieu training goc"""
    return split_raw_json_data(
        config.TRAIN_JSON_PATH,
        config.TRAIN_SPLIT_JSON_PATH,
        config.VAL_SPLIT_JSON_PATH,
        val_size=0.1,
        random_state=42,
    )


def split_triplets_data():
    """Chia du lieu triplets"""
    return split_jsonl_data(
        config.TRAIN_TRIPLETS_EASY_PATH,
        config.TRAIN_TRIPLETS_EASY_FOR_TRAINING_PATH,
        config.BI_ENCODER_VALIDATION_PATH,
        val_size=0.15,
        random_state=42,
    )


def main():
    """Ham chinh"""
    logging.info("[START] Bat dau chia du lieu...")

    try:
        # Chia du lieu training goc
        if not split_raw_training_data():
            logging.error("[FAIL] Loi khi chia du lieu training goc")
            return False

        # Chia du lieu triplets (neu co)
        if config.TRAIN_TRIPLETS_EASY_PATH.exists():
            if not split_triplets_data():
                logging.error("[FAIL] Loi khi chia du lieu triplets")
                return False

        logging.info("[SUCCESS] Tat ca du lieu da duoc chia thanh cong!")
        return True

    except Exception as e:
        logging.error(f"[FAIL] Loi khi chia du lieu: {e}")
        return False


if __name__ == "__main__":
    main()
