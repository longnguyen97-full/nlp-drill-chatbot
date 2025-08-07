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
import time
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
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"[INFO] GPU Memory: {gpu_memory_gb:.1f} GB")

            if gpu_memory_gb >= 8:
                logger.info("[OK] GPU memory sufficient for optimal training")
            elif gpu_memory_gb >= 4:
                logger.info(
                    "[WARN] GPU memory moderate (may need to reduce batch size)"
                )
            else:
                logger.info("[WARN] GPU memory low (will use smaller batch sizes)")
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

    # Check if processed data exists
    if config.AID_MAP_PATH.exists():
        size_mb = config.AID_MAP_PATH.stat().st_size / (1024 * 1024)
        logger.info(f"[OK] AID Map: {size_mb:.1f} MB")
    else:
        logger.info("[INFO] AID Map: Will be created in this step")

    if config.TRAIN_SPLIT_JSON_PATH.exists():
        size_mb = config.TRAIN_SPLIT_JSON_PATH.stat().st_size / (1024 * 1024)
        logger.info(f"[OK] Train Split: {size_mb:.1f} MB")
    else:
        logger.info("[INFO] Train Split: Will be created in this step")

    if config.VAL_SPLIT_JSON_PATH.exists():
        size_mb = config.VAL_SPLIT_JSON_PATH.stat().st_size / (1024 * 1024)
        logger.info(f"[OK] Validation Split: {size_mb:.1f} MB")
    else:
        logger.info("[INFO] Validation Split: Will be created in this step")

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


def check_configuration():
    """Kiem tra cau hinh"""
    logger.info("[CONFIG] Kiem tra cau hinh...")

    try:
        config.validate_config()
        logger.info("[OK] Configuration validation passed")
    except Exception as e:
        logger.error(f"[FAIL] Configuration validation failed: {e}")
        return False

    # Log hyperparameters
    logger.info(f"[INFO] Bi-Encoder Batch Size: {config.BI_ENCODER_BATCH_SIZE}")
    logger.info(f"[INFO] Bi-Encoder Epochs: {config.BI_ENCODER_EPOCHS}")
    logger.info(f"[INFO] Cross-Encoder Batch Size: {config.CROSS_ENCODER_BATCH_SIZE}")
    logger.info(f"[INFO] Cross-Encoder Epochs: {config.CROSS_ENCODER_EPOCHS}")
    logger.info(f"[INFO] Learning Rate: {config.BI_ENCODER_LR}")

    return True


def check_existing_models():
    """Kiem tra cac model da ton tai"""
    logger.info("[MODELS] Kiem tra cac model da ton tai...")

    models = [
        (config.PHOBERT_LAW_PATH, "PhoBERT-Law"),
        (config.BI_ENCODER_PATH, "Bi-Encoder"),
        (config.CROSS_ENCODER_PATH, "Cross-Encoder"),
        (config.LIGHT_RERANKER_PATH, "Light Reranker"),
        (config.FAISS_INDEX_PATH, "FAISS Index"),
        (config.INDEX_TO_AID_PATH, "Index to AID Mapping"),
    ]

    existing_models = []
    for model_path, name in models:
        if model_path.exists():
            if model_path.is_file():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                logger.info(f"[WARN] {name}: {size_mb:.1f} MB (will be replaced)")
                existing_models.append(name)
            else:
                logger.info(f"[WARN] {name}: Directory exists (will be replaced)")
                existing_models.append(name)
        else:
            logger.info(f"[OK] {name}: Not found (will be created)")

    if existing_models:
        logger.info(
            f"[WARN] Found {len(existing_models)} existing models that will be replaced"
        )
        logger.info(f"[WARN] Models: {', '.join(existing_models)}")

    return True


def build_maps_optimized():
    """Build maps voi fix mapping issues"""
    logger.info("[MAP] Building optimized maps with mapping fixes...")

    # Load legal corpus
    logger.info("[MAP] Loading legal corpus...")
    with open(config.LEGAL_CORPUS_PATH, "r", encoding="utf-8") as f:
        legal_corpus = json.load(f)
    logger.info(f"[MAP] Loaded {len(legal_corpus)} legal documents")

    # Load training data
    logger.info("[MAP] Loading training data...")
    with open(config.TRAIN_JSON_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    logger.info(f"[MAP] Loaded {len(train_data)} training samples")

    # Build aid_map (AID -> content)
    logger.info("[MAP] Building AID mappings...")
    aid_map = {}
    doc_id_to_aids = {}
    total_articles = 0

    for i, doc in enumerate(legal_corpus):
        if i % 1000 == 0:
            logger.info(f"[MAP] Processing document {i+1}/{len(legal_corpus)}...")

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
            total_articles += 1

    logger.info(
        f"[MAP] Created {len(aid_map)} AID mappings from {total_articles} articles"
    )

    # Fix training data mapping issues
    logger.info("[MAP] Fixing training data mapping issues...")
    fixed_train_data = []
    mapping_fixes = 0

    for i, sample in enumerate(train_data):
        if i % 100 == 0:
            logger.info(f"[MAP] Processing training sample {i+1}/{len(train_data)}...")

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
            if len(valid_aids) > 0:
                mapping_fixes += 1

    logger.info(f"[MAP] Fixed {mapping_fixes} training samples with valid AIDs")

    # Save fixed training data
    logger.info("[MAP] Saving processed data...")
    train_fixed_path = config.DATA_RAW_DIR / "train_fixed.json"
    with open(train_fixed_path, "w", encoding="utf-8") as f:
        json.dump(fixed_train_data, f, ensure_ascii=False, indent=2)
    logger.info(f"[MAP] Saved fixed training data to {train_fixed_path}")

    # Save maps
    with open(config.AID_MAP_PATH, "wb") as f:
        pickle.dump(aid_map, f)
    logger.info(f"[MAP] Saved AID map to {config.AID_MAP_PATH}")

    with open(config.DOC_ID_TO_AIDS_PATH, "w", encoding="utf-8") as f:
        json.dump(doc_id_to_aids, f, ensure_ascii=False, indent=2)
    logger.info(f"[MAP] Saved doc_id_to_aids mapping to {config.DOC_ID_TO_AIDS_PATH}")

    # Create evaluation mapping info
    evaluation_mapping = {
        "train_samples": len(fixed_train_data),
        "total_aids": len(aid_map),
        "total_docs": len(doc_id_to_aids),
        "total_articles": total_articles,
        "mapping_fixes": mapping_fixes,
        "format": "aid_based",
        "mapping_fixed": True,
    }

    evaluation_mapping_path = config.DATA_PROCESSED_DIR / "evaluation_mapping.json"
    with open(evaluation_mapping_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_mapping, f, ensure_ascii=False, indent=2)
    logger.info(f"[MAP] Saved evaluation mapping info to {evaluation_mapping_path}")

    logger.info(f"[MAP] Successfully created {len(aid_map)} AIDs")
    logger.info(f"[MAP] Successfully created {len(doc_id_to_aids)} doc mappings")
    logger.info(f"[MAP] Successfully fixed {len(fixed_train_data)} training samples")
    logger.info(f"[MAP] Mapping issues fixed successfully!")

    return aid_map, doc_id_to_aids, fixed_train_data


def split_data_optimized(train_data):
    """Chia du lieu toi uu"""
    logger.info("[SPLIT] Splitting data with optimized logic...")

    # Set random seed for reproducibility
    random.seed(42)
    logger.info("[SPLIT] Set random seed to 42 for reproducibility")

    # Shuffle data
    logger.info("[SPLIT] Shuffling training data...")
    random.shuffle(train_data)
    logger.info("[SPLIT] Data shuffled successfully")

    # Calculate split point
    train_ratio = 0.85
    train_count = int(len(train_data) * train_ratio)
    val_count = len(train_data) - train_count

    logger.info(f"[SPLIT] Split configuration:")
    logger.info(f"[SPLIT]   • Total samples: {len(train_data)}")
    logger.info(f"[SPLIT]   • Train ratio: {train_ratio:.1%}")
    logger.info(f"[SPLIT]   • Train samples: {train_count}")
    logger.info(f"[SPLIT]   • Validation samples: {val_count}")
    logger.info(f"[SPLIT]   • Validation ratio: {val_count/len(train_data):.1%}")

    # Split data (FIXED logic)
    logger.info("[SPLIT] Performing data split...")
    train_split = train_data[:train_count]
    val_split = train_data[train_count:]
    logger.info("[SPLIT] Data split completed")

    # Validate split
    logger.info("[SPLIT] Validating split results...")
    assert len(train_split) + len(val_split) == len(
        train_data
    ), "Split validation failed"
    logger.info("[SPLIT] Split validation passed")

    # Save splits
    logger.info("[SPLIT] Saving split data...")
    train_split_path = config.TRAIN_SPLIT_JSON_PATH
    val_split_path = config.VAL_SPLIT_JSON_PATH

    with open(train_split_path, "w", encoding="utf-8") as f:
        json.dump(train_split, f, ensure_ascii=False, indent=2)
    logger.info(f"[SPLIT] Training data saved to: {train_split_path}")

    with open(val_split_path, "w", encoding="utf-8") as f:
        json.dump(val_split, f, ensure_ascii=False, indent=2)
    logger.info(f"[SPLIT] Validation data saved to: {val_split_path}")

    # Log sample statistics
    logger.info("[SPLIT] Sample statistics:")
    logger.info(f"[SPLIT]   • Training set size: {len(train_split)} samples")
    logger.info(f"[SPLIT]   • Validation set size: {len(val_split)} samples")
    logger.info(
        f"[SPLIT]   • Total AIDs in training: {sum(len(sample.get('relevant_aids', [])) for sample in train_split)}"
    )
    logger.info(
        f"[SPLIT]   • Total AIDs in validation: {sum(len(sample.get('relevant_aids', [])) for sample in val_split)}"
    )

    logger.info("[SPLIT] Data splitting completed successfully!")

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
        logger.info("[VALID] All validation AIDs exist in aid_map!")

    if validation_rate >= 0.95:
        logger.info("[VALID] Mapping validation passed!")
        return True
    else:
        logger.error(f"[VALID] Mapping validation failed! Rate: {validation_rate:.2%}")
        return False


def run_complete_pipeline():
    """Chay pipeline hoan chinh: Environment + Data Processing + Validation + Comprehensive Readiness Check"""
    logger.info("=" * 80)
    logger.info("STARTING: ENVIRONMENT & DATA PROCESSING & READINESS CHECK PIPELINE")
    logger.info("=" * 80)
    logger.info("Pipeline Overview:")
    logger.info("   - Step 1: Environment Check (Python, PyTorch, CUDA, Libraries)")
    logger.info("   - Step 2: Configuration Validation (Hyperparameters, Settings)")
    logger.info("   - Step 3: Model Status Check (Existing Models Detection)")
    logger.info("   - Step 4: Data Files Validation (Input & Processed Files)")
    logger.info("   - Step 5: Directory Structure Setup")
    logger.info("   - Step 6: Data Processing & Mapping (AID Maps, Training Data)")
    logger.info("   - Step 7: Data Splitting (Train/Validation Sets)")
    logger.info("   - Step 8: Mapping Validation (AID Consistency Check)")
    logger.info("   - Step 9: Final Readiness Summary")
    logger.info("=" * 80)

    start_time = time.time()
    step_results = []

    try:
        # Step 1: Check environment
        logger.info("")
        logger.info("STEP 1: ENVIRONMENT CHECK")
        logger.info("-" * 40)
        step_start = time.time()
        if not check_environment():
            logger.error("STEP 1 FAILED: Environment check failed")
            return False
        step_time = time.time() - step_start
        logger.info(f"STEP 1 COMPLETED: Environment check passed ({step_time:.2f}s)")
        step_results.append(("Environment Check", True, step_time))

        # Step 2: Check configuration
        logger.info("")
        logger.info("STEP 2: CONFIGURATION VALIDATION")
        logger.info("-" * 40)
        step_start = time.time()
        if not check_configuration():
            logger.error("STEP 2 FAILED: Configuration validation failed")
            return False
        step_time = time.time() - step_start
        logger.info(f"STEP 2 COMPLETED: Configuration validated ({step_time:.2f}s)")
        step_results.append(("Configuration Validation", True, step_time))

        # Step 3: Check existing models
        logger.info("")
        logger.info("STEP 3: MODEL STATUS CHECK")
        logger.info("-" * 40)
        step_start = time.time()
        if not check_existing_models():
            logger.error("STEP 3 FAILED: Model status check failed")
            return False
        step_time = time.time() - step_start
        logger.info(f"STEP 3 COMPLETED: Model status checked ({step_time:.2f}s)")
        step_results.append(("Model Status Check", True, step_time))

        # Step 4: Check data files
        logger.info("")
        logger.info("STEP 4: DATA FILES VALIDATION")
        logger.info("-" * 40)
        step_start = time.time()
        if not check_data_files():
            logger.error("STEP 4 FAILED: Data files validation failed")
            return False
        step_time = time.time() - step_start
        logger.info(f"STEP 4 COMPLETED: Data files validated ({step_time:.2f}s)")
        step_results.append(("Data Files Validation", True, step_time))

        # Step 5: Create output directories
        logger.info("")
        logger.info("STEP 5: DIRECTORY STRUCTURE SETUP")
        logger.info("-" * 40)
        step_start = time.time()
        if not create_output_dirs():
            logger.error("STEP 5 FAILED: Directory creation failed")
            return False
        step_time = time.time() - step_start
        logger.info(f"STEP 5 COMPLETED: Directories created ({step_time:.2f}s)")
        step_results.append(("Directory Setup", True, step_time))

        # Step 6: Build maps and fix training data
        logger.info("")
        logger.info("STEP 6: DATA PROCESSING & MAPPING")
        logger.info("-" * 40)
        step_start = time.time()
        aid_map, doc_id_to_aids, fixed_train_data = build_maps_optimized()
        step_time = time.time() - step_start
        logger.info(f"STEP 6 COMPLETED: Data processing finished ({step_time:.2f}s)")
        logger.info(f"   - Created {len(aid_map)} AID mappings")
        logger.info(f"   - Processed {len(fixed_train_data)} training samples")
        step_results.append(("Data Processing", True, step_time))

        # Step 7: Split data
        logger.info("")
        logger.info("STEP 7: DATA SPLITTING")
        logger.info("-" * 40)
        step_start = time.time()
        train_split, val_split = split_data_optimized(fixed_train_data)
        step_time = time.time() - step_start
        logger.info(f"STEP 7 COMPLETED: Data splitting finished ({step_time:.2f}s)")
        logger.info(f"   - Training set: {len(train_split)} samples")
        logger.info(f"   - Validation set: {len(val_split)} samples")
        step_results.append(("Data Splitting", True, step_time))

        # Step 8: Validate mapping
        logger.info("")
        logger.info("STEP 8: MAPPING VALIDATION")
        logger.info("-" * 40)
        step_start = time.time()
        validation_success = validate_mapping_optimized(aid_map, val_split)
        if not validation_success:
            logger.error("STEP 8 FAILED: Mapping validation failed")
            return False
        step_time = time.time() - step_start
        logger.info(f"STEP 8 COMPLETED: Mapping validation passed ({step_time:.2f}s)")
        step_results.append(("Mapping Validation", True, step_time))

        # Step 9: Final readiness summary
        total_time = time.time() - start_time
        logger.info("")
        logger.info("=" * 80)
        logger.info("FINAL READINESS CHECK SUMMARY")
        logger.info("=" * 80)
        logger.info("SYSTEM IS READY FOR TRAINING!")
        logger.info("")
        logger.info("STEP-BY-STEP RESULTS:")
        for step_name, success, step_time in step_results:
            status = "PASSED" if success else "FAILED"
            logger.info(f"   {status} {step_name} ({step_time:.2f}s)")

        logger.info("")
        logger.info("ENVIRONMENT STATUS:")
        logger.info("   - All environment checks passed")
        logger.info("   - All data files validated")
        logger.info("   - Configuration validated")
        logger.info("   - Model status checked")
        logger.info("   - Data processing completed")
        logger.info("   - Mapping validation passed")

        logger.info("")
        logger.info("DATA SUMMARY:")
        logger.info(f"   - Total AIDs: {len(aid_map)}")
        logger.info(f"   - Training samples: {len(train_split)}")
        logger.info(f"   - Validation samples: {len(val_split)}")
        logger.info(
            f"   - Validation rate: {len(val_split)/len(fixed_train_data)*100:.1f}%"
        )

        logger.info("")
        logger.info("PERFORMANCE SUMMARY:")
        logger.info(f"   - Total execution time: {total_time:.2f} seconds")
        logger.info(
            f"   - Average step time: {total_time/len(step_results):.2f} seconds"
        )

        logger.info("")
        logger.info("NEXT STEPS:")
        logger.info("   1. Run training data preparation pipeline (Step 02)")
        logger.info("   2. Run model training pipeline (Step 03)")
        logger.info("   3. Deploy system")

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        return True

    except Exception as e:
        logger.error("=" * 80)
        logger.error("PIPELINE FAILED WITH EXCEPTION")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error("Please check the logs above for details.")
        logger.error("=" * 80)
        return False


def main():
    """Ham chinh"""
    logger.info("[START] Bat dau Complete Data & Environment Pipeline...")

    success = run_complete_pipeline()

    if success:
        logger.info("Complete Data & Environment Pipeline completed successfully!")
        logger.info("Environment and data ready for training!")
        return True
    else:
        logger.error("Complete Data & Environment Pipeline failed!")
        return False


if __name__ == "__main__":
    main()
