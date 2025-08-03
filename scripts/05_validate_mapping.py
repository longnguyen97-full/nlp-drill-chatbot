#!/usr/bin/env python3
"""
Validate mapping giua doc_ids va AIDs
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


def validate_validation_mapping():
    """Validate mapping cho validation set"""
    logging.info("[SEARCH] Kiem tra mapping cho validation set...")

    # Load validation data
    with open(config.VAL_SPLIT_JSON_PATH, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    # Load doc_id_to_aids mapping
    with open(config.DOC_ID_TO_AIDS_PATH, "r", encoding="utf-8") as f:
        doc_id_to_aids = json.load(f)

    # Kiem tra tung mau trong validation set
    missing_mappings = []
    total_samples = len(val_data)

    for i, sample in enumerate(val_data):
        if i % 1000 == 0:
            logging.info(f"[PROGRESS] Da kiem tra {i}/{total_samples} mau...")

        # Lay doc_ids tu positive_ctxs
        positive_doc_ids = []
        for ctx in sample.get("positive_ctxs", []):
            doc_id = ctx.get("doc_id")
            if doc_id is not None:
                positive_doc_ids.append(doc_id)

        # Kiem tra xem tat ca doc_ids co mapping khong
        for doc_id in positive_doc_ids:
            if str(doc_id) not in doc_id_to_aids:
                missing_mappings.append(
                    {
                        "sample_index": i,
                        "doc_id": doc_id,
                        "question": sample.get("question", "")[:100],
                    }
                )

    if missing_mappings:
        logging.error(
            f"[FAIL] Tim thay {len(missing_mappings)} doc_ids khong co mapping:"
        )
        for missing in missing_mappings[:10]:  # Chi hien thi 10 dau tien
            logging.error(
                f"  - Sample {missing['sample_index']}, doc_id {missing['doc_id']}"
            )

        if len(missing_mappings) > 10:
            logging.error(f"  ... va {len(missing_mappings) - 10} mau khac")

        return False
    else:
        logging.info("[OK] All validation doc_ids have proper mapping!")
        return True


def create_complete_mapping():
    """Tao mapping hoan chinh cho tat ca doc_ids"""
    logging.info("[BUILD] Tao mapping hoan chinh...")

    # Load legal corpus
    with open(config.LEGAL_CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    # Tao mapping hoan chinh
    complete_mapping = {}
    for doc in corpus_data:
        doc_id = doc.get("doc_id")
        if doc_id is not None:
            complete_mapping[str(doc_id)] = doc.get("aid", "")

    # Luu mapping hoan chinh
    complete_mapping_path = config.DATA_PROCESSED_DIR / "doc_id_to_aid_complete.json"
    with open(complete_mapping_path, "w", encoding="utf-8") as f:
        json.dump(complete_mapping, f, ensure_ascii=False, indent=2)

    logging.info(f"[FILE] Complete mapping saved to: {complete_mapping_path}")
    logging.info(f"[INFO] Total mappings: {len(complete_mapping)}")

    # Cap nhat config
    logging.info("[NOTE] Update config.py to use complete mapping:")
    logging.info(
        f"DOC_ID_TO_AIDS_PATH = DATA_PROCESSED_DIR / 'doc_id_to_aid_complete.json'"
    )

    return complete_mapping_path


def main():
    """Ham chinh"""
    logging.info("[START] Bat dau validate mapping...")

    try:
        # Kiem tra mapping hien tai
        if validate_validation_mapping():
            logging.info("[SUCCESS] Mapping validation passed!")
            return True
        else:
            logging.warning(
                "[WARNING] Mapping validation failed, creating complete mapping..."
            )

            # Tao mapping hoan chinh
            complete_mapping_path = create_complete_mapping()

            # Cap nhat config de su dung mapping moi
            config.DOC_ID_TO_AIDS_PATH = complete_mapping_path

            # Kiem tra lai
            if validate_validation_mapping():
                logging.info("[SUCCESS] Complete mapping validation passed!")
                return True
            else:
                logging.error("[FAIL] Complete mapping validation still failed!")
                return False

    except Exception as e:
        logging.error(f"[FAIL] Loi khi validate mapping: {e}")
        return False


if __name__ == "__main__":
    main()
