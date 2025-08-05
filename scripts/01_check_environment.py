#!/usr/bin/env python3
"""
Complete Data & Environment Pipeline - Script Toi Uu Cuc Dai
=========================================================

Script nay kiem tra moi truong, xu ly du lieu, validation, va chuan bi training data
trong mot buoc toi uu cuc dai de tang hieu qua toi da.

Tac gia: LawBot Team
Phien ban: Maximum Optimized Pipeline v5.0
"""

import sys
import os
import json
import pickle
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

# Them thu muc goc vao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from core.logging_system import get_logger

# Sử dụng logger đã được setup từ pipeline chính
logger = get_logger(__name__)


def check_environment():
    """Kiem tra moi truong Python va dependencies"""
    logger.info("[SEARCH] Kiem tra moi truong Python...")

    # Kiem tra Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        logger.info(
            f"[OK] Python version: {python_version.major}.{python_version.minor}.{python_version.micro}"
        )
    else:
        logger.error(
            f"[FAIL] Python version {python_version.major}.{python_version.minor} khong du. Can Python 3.8+"
        )
        return False

    # Kiem tra PyTorch
    try:
        import torch

        logger.info(f"[OK] PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("[NOTE] CUDA khong kha dung, se su dung CPU")
    except ImportError:
        logger.error("[FAIL] PyTorch chua duoc cai dat")
        return False

    # Kiem tra cac thu vien khac
    required_libs = [
        "transformers",
        "sentence_transformers",
        "faiss",
        "numpy",
        "pandas",
        "sklearn",
        "streamlit",
    ]

    for lib in required_libs:
        try:
            __import__(lib)
            logger.info(f"[OK] {lib} da duoc cai dat")
        except ImportError:
            logger.error(f"[FAIL] {lib} chua duoc cai dat")
            return False

    return True


def check_data_files():
    """Kiem tra cac file du lieu can thiet"""
    logger.info("[FILE] Kiem tra file du lieu...")

    required_files = [
        config.TRAIN_JSON_PATH,
        config.LEGAL_CORPUS_PATH,
        config.PUBLIC_TEST_JSON_PATH,
    ]

    for file_path in required_files:
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"[OK] {file_path.name}: {size_mb:.2f} MB")
        else:
            logger.error(f"[FAIL] Thieu file: {file_path}")
            return False

    return True


def create_output_dirs():
    """Tao thu muc output neu chua co"""
    logger.info("[DIR] Tao thu muc output...")

    dirs = [
        config.DATA_PROCESSED_DIR,
        config.DATA_VALIDATION_DIR,
        config.MODELS_DIR,
        config.INDEXES_DIR,
        config.REPORTS_DIR,
        Path("logs"),
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    logger.info("[OK] Cac thu muc da duoc tao")
    return True


def build_maps_optimized():
    """Build maps voi fix mapping issues"""
    logger.info("[MAP] Building optimized maps with mapping fixes...")

    # Load legal corpus
    with open(config.LEGAL_CORPUS_PATH, "r", encoding="utf-8") as f:
        legal_corpus = json.load(f)

    # Load training data
    with open(config.TRAIN_JSON_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # Build aid_map (AID -> content)
    aid_map = {}
    doc_id_to_aids = {}

    logger.info(f"[MAP] Processing {len(legal_corpus)} legal documents...")

    for doc in legal_corpus:
        doc_id = doc["id"]
        law_id = doc["law_id"]
        content_items = doc.get("content", [])

        # Convert doc_id to string for consistency
        doc_id_str = str(doc_id)

        if doc_id_str not in doc_id_to_aids:
            doc_id_to_aids[doc_id_str] = []

        for item in content_items:
            article_id = item["aid"]
            content = item["content_Article"]

            # Create AID format: law_id_article_id
            aid = f"{law_id}_{article_id}"

            # Store in aid_map
            aid_map[aid] = content

            # Store mapping from doc_id to AIDs
            doc_id_to_aids[doc_id_str].append(aid)

    # Fix training data mapping issues
    logger.info("[MAP] Fixing training data mapping issues...")
    fixed_train_data = []

    for sample in train_data:
        doc_ids = sample["relevant_laws"]
        valid_aids = []

        # Convert doc_ids to AIDs
        for doc_id in doc_ids:
            doc_id_str = str(doc_id)
            if doc_id_str in doc_id_to_aids:
                aids = doc_id_to_aids[doc_id_str]
                valid_aids.extend(aids)

        if valid_aids:
            # Create fixed sample
            fixed_sample = {
                "qid": sample["qid"],
                "question": sample["question"],
                "relevant_aids": valid_aids,  # Changed from relevant_laws
            }
            fixed_train_data.append(fixed_sample)

    # Save fixed training data
    train_fixed_path = config.DATA_RAW_DIR / "train_fixed.json"
    with open(train_fixed_path, "w", encoding="utf-8") as f:
        json.dump(fixed_train_data, f, ensure_ascii=False, indent=2)

    # Save maps
    with open(config.AID_MAP_PATH, "wb") as f:
        pickle.dump(aid_map, f)

    with open(config.DOC_ID_TO_AIDS_PATH, "w", encoding="utf-8") as f:
        json.dump(doc_id_to_aids, f, ensure_ascii=False, indent=2)

    # Create evaluation mapping info
    evaluation_mapping = {
        "train_samples": len(fixed_train_data),
        "total_aids": len(aid_map),
        "total_docs": len(doc_id_to_aids),
        "format": "aid_based",
        "mapping_fixed": True,
    }

    evaluation_mapping_path = config.DATA_PROCESSED_DIR / "evaluation_mapping.json"
    with open(evaluation_mapping_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_mapping, f, ensure_ascii=False, indent=2)

    logger.info(f"[MAP] Created {len(aid_map)} AIDs")
    logger.info(f"[MAP] Created {len(doc_id_to_aids)} doc mappings")
    logger.info(f"[MAP] Fixed {len(fixed_train_data)} training samples")
    logger.info("[MAP] Mapping issues fixed successfully!")

    return aid_map, doc_id_to_aids, fixed_train_data


def split_data_optimized(train_data):
    """Chia du lieu toi uu"""
    logger.info("[SPLIT] Splitting data with optimized logic...")

    # Set random seed
    random.seed(42)

    # Shuffle data
    random.shuffle(train_data)

    # Calculate split point
    train_ratio = 0.85
    train_count = int(len(train_data) * train_ratio)
    val_count = len(train_data) - train_count

    logger.info(f"[SPLIT] Total samples: {len(train_data)}")
    logger.info(f"[SPLIT] Train samples: {train_count}")
    logger.info(f"[SPLIT] Validation samples: {val_count}")

    # Split data (FIXED logic)
    train_split = train_data[:train_count]
    val_split = train_data[train_count:]

    # Save splits
    train_split_path = config.TRAIN_SPLIT_JSON_PATH
    val_split_path = config.VAL_SPLIT_JSON_PATH

    with open(train_split_path, "w", encoding="utf-8") as f:
        json.dump(train_split, f, ensure_ascii=False, indent=2)

    with open(val_split_path, "w", encoding="utf-8") as f:
        json.dump(val_split, f, ensure_ascii=False, indent=2)

    logger.info(f"[SPLIT] Training data saved to: {train_split_path}")
    logger.info(f"[SPLIT] Validation data saved to: {val_split_path}")

    return train_split, val_split


def validate_mapping_optimized(aid_map, val_data):
    """Validate mapping toi uu"""
    logger.info("[VALID] Validating mapping with optimized logic...")

    # Check if all AIDs in validation set exist in aid_map
    missing_aids = []
    valid_samples = 0

    for sample in val_data:
        relevant_aids = sample.get("relevant_aids", [])
        all_valid = True

        for aid in relevant_aids:
            if aid not in aid_map:
                missing_aids.append(aid)
                all_valid = False

        if all_valid:
            valid_samples += 1

    total_samples = len(val_data)
    validation_rate = valid_samples / total_samples if total_samples > 0 else 0

    logger.info(f"[VALID] Validation samples: {total_samples}")
    logger.info(f"[VALID] Valid samples: {valid_samples}")
    logger.info(f"[VALID] Validation rate: {validation_rate:.2%}")

    if missing_aids:
        logger.warning(f"[VALID] Missing AIDs: {len(missing_aids)}")
        logger.warning(f"[VALID] Missing AIDs: {missing_aids[:10]}...")  # Show first 10
    else:
        logger.info("[VALID] ✅ All validation AIDs exist in aid_map!")

    if validation_rate >= 0.95:
        logger.info("[VALID] ✅ Mapping validation passed!")
        return True
    else:
        logger.error(
            f"[VALID] ❌ Mapping validation failed! Rate: {validation_rate:.2%}"
        )
        return False


def run_complete_pipeline():
    """Chay pipeline hoan chinh: Environment + Data Processing + Validation"""
    logger.info("=" * 60)
    logger.info("ENVIRONMENT & DATA PROCESSING PIPELINE")
    logger.info("=" * 60)

    try:
        # Step 1: Check environment
        logger.info("STEP 1: Checking environment...")
        if not check_environment():
            return False

        # Step 2: Check data files
        logger.info("STEP 2: Checking data files...")
        if not check_data_files():
            return False

        # Step 3: Create output directories
        logger.info("STEP 3: Creating output directories...")
        if not create_output_dirs():
            return False

        # Step 4: Build maps and fix training data
        logger.info("STEP 4: Building maps and fixing training data...")
        aid_map, doc_id_to_aids, fixed_train_data = build_maps_optimized()

        # Step 5: Split data
        logger.info("STEP 5: Splitting data...")
        train_split, val_split = split_data_optimized(fixed_train_data)

        # Step 6: Validate mapping
        logger.info("STEP 6: Validating mapping...")
        validation_success = validate_mapping_optimized(aid_map, val_split)

        if not validation_success:
            logger.error("Pipeline failed due to mapping validation issues!")
            return False

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Next steps:")
        logger.info("1. Run training data preparation pipeline")
        logger.info("2. Run model training pipeline")
        logger.info("3. Deploy system")
        return True

    except Exception as e:
        logger.error(f"Error during pipeline: {e}")
        return False


def main():
    """Ham chinh"""
    logger.info("[START] Bat dau Complete Data & Environment Pipeline...")

    success = run_complete_pipeline()

    if success:
        logger.info("✅ Complete Data & Environment Pipeline completed successfully!")
        logger.info("✅ Moi truong va du lieu san sang cho training!")
    else:
        logger.error("❌ Complete Data & Environment Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
