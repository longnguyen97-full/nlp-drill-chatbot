#!/usr/bin/env python3
"""
Base Configuration for LawBot
============================

This file contains common configuration settings and utility functions
that are shared across all performance modes.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# ============================================================================
# ENVIRONMENT VARIABLES SUPPORT WITH ENHANCED ERROR HANDLING
# ============================================================================


def get_env_var(key: str, default: str = None) -> Optional[str]:
    """Get environment variable with fallback to default"""
    value = os.getenv(key, default)
    if value is None and default is None:
        logging.warning(f"Environment variable {key} not set and no default provided")
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable with validation"""
    value = os.getenv(key, str(default)).lower()
    valid_true_values = ("true", "1", "yes", "on")
    valid_false_values = ("false", "0", "no", "off")

    if value in valid_true_values:
        return True
    elif value in valid_false_values:
        return False
    else:
        logging.warning(
            f"Invalid boolean value '{value}' for {key}, using default: {default}"
        )
        return default


def get_env_int(
    key: str, default: int = 0, min_value: int = None, max_value: int = None
) -> int:
    """Get integer environment variable with validation"""
    try:
        value = int(os.getenv(key, str(default)))

        # Validate range if specified
        if min_value is not None and value < min_value:
            logging.warning(
                f"Value {value} for {key} is below minimum {min_value}, using {min_value}"
            )
            value = min_value
        if max_value is not None and value > max_value:
            logging.warning(
                f"Value {value} for {key} is above maximum {max_value}, using {max_value}"
            )
            value = max_value

        return value
    except ValueError:
        logging.error(f"Invalid integer value for {key}, using default: {default}")
        return default


def get_env_float(
    key: str, default: float = 0.0, min_value: float = None, max_value: float = None
) -> float:
    """Get float environment variable with validation"""
    try:
        value = float(os.getenv(key, str(default)))

        # Validate range if specified
        if min_value is not None and value < min_value:
            logging.warning(
                f"Value {value} for {key} is below minimum {min_value}, using {min_value}"
            )
            value = min_value
        if max_value is not None and value > max_value:
            logging.warning(
                f"Value {value} for {key} is above maximum {max_value}, using {max_value}"
            )
            value = max_value

        return value
    except ValueError:
        logging.error(f"Invalid float value for {key}, using default: {default}")
        return default


# ============================================================================
# BASE CONFIGURATION WITH ENHANCED VALIDATION
# ============================================================================

# Thu muc goc cua du an
ROOT_DIR = Path(__file__).parent

# Environment
ENVIRONMENT = get_env_var("LAWBOT_ENV", "development")
DEBUG = get_env_bool("LAWBOT_DEBUG", True)

# ============================================================================
# DIRECTORY CONFIGURATION
# ============================================================================

# --- Thu muc chinh ---
DATA_DIR = Path(get_env_var("LAWBOT_DATA_DIR", str(ROOT_DIR / "data")))
MODELS_DIR = Path(get_env_var("LAWBOT_MODELS_DIR", str(ROOT_DIR / "models")))
INDEXES_DIR = Path(get_env_var("LAWBOT_INDEXES_DIR", str(ROOT_DIR / "indexes")))
REPORTS_DIR = Path(get_env_var("LAWBOT_REPORTS_DIR", str(ROOT_DIR / "reports")))
LOGS_DIR = Path(get_env_var("LAWBOT_LOGS_DIR", str(ROOT_DIR / "logs")))

# --- Duong dan Du lieu ---
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_VALIDATION_DIR = DATA_DIR / "validation"

# Input files
LEGAL_CORPUS_PATH = Path(
    get_env_var("LAWBOT_LEGAL_CORPUS_PATH", str(DATA_RAW_DIR / "legal_corpus.json"))
)
TRAIN_JSON_PATH = Path(
    get_env_var("LAWBOT_TRAIN_JSON_PATH", str(DATA_RAW_DIR / "train.json"))
)
PUBLIC_TEST_JSON_PATH = Path(
    get_env_var("LAWBOT_PUBLIC_TEST_PATH", str(DATA_RAW_DIR / "public_test.json"))
)

# Enhanced data collection files
TRAIN_EXTENDED_PATH = Path(
    get_env_var("LAWBOT_TRAIN_EXTENDED_PATH", str(DATA_RAW_DIR / "train_extended.json"))
)
TRAIN_ENHANCED_AUGMENTED_PATH = Path(
    get_env_var(
        "LAWBOT_TRAIN_ENHANCED_AUGMENTED_PATH",
        str(DATA_RAW_DIR / "train_enhanced_augmented.json"),
    )
)
TRAIN_EXTENDED_ENHANCED_AUGMENTED_PATH = Path(
    get_env_var(
        "LAWBOT_TRAIN_EXTENDED_ENHANCED_AUGMENTED_PATH",
        str(DATA_RAW_DIR / "train_extended_enhanced_augmented.json"),
    )
)

# Du lieu duoc chia de huan luyen va danh gia cuoi cung
TRAIN_SPLIT_JSON_PATH = Path(
    get_env_var("LAWBOT_TRAIN_SPLIT_PATH", str(DATA_RAW_DIR / "train_split.json"))
)
VAL_SPLIT_JSON_PATH = Path(
    get_env_var("LAWBOT_VAL_SPLIT_PATH", str(DATA_RAW_DIR / "validation_split.json"))
)

# Processed files
AID_MAP_PATH = Path(
    get_env_var("LAWBOT_AID_MAP_PATH", str(DATA_PROCESSED_DIR / "aid_map.pkl"))
)
DOC_ID_TO_AIDS_PATH = Path(
    get_env_var(
        "LAWBOT_DOC_ID_TO_AIDS_PATH",
        str(DATA_PROCESSED_DIR / "doc_id_to_aids_complete.json"),
    )
)

# Du lieu triplets "easy" duoc tao tu train_split.json
TRAIN_TRIPLETS_EASY_PATH = Path(
    get_env_var(
        "LAWBOT_TRAIN_TRIPLETS_EASY_PATH",
        str(DATA_PROCESSED_DIR / "train_triplets_easy.jsonl"),
    )
)
# Phan training cua du lieu "easy"
TRAIN_TRIPLETS_EASY_FOR_TRAINING_PATH = Path(
    get_env_var(
        "LAWBOT_TRAIN_TRIPLETS_EASY_FOR_TRAINING_PATH",
        str(DATA_PROCESSED_DIR / "train_triplets_easy_for_training.jsonl"),
    )
)
# Du lieu validation cho Bi-Encoder (tach tu tap easy)
BI_ENCODER_VALIDATION_PATH = Path(
    get_env_var(
        "LAWBOT_BI_ENCODER_VALIDATION_PATH",
        str(DATA_PROCESSED_DIR / "bi_encoder_validation.jsonl"),
    )
)
# Du lieu triplets "hard" negatives
TRAIN_TRIPLETS_HARD_NEG_PATH = Path(
    get_env_var(
        "LAWBOT_TRAIN_TRIPLETS_HARD_NEG_PATH",
        str(DATA_PROCESSED_DIR / "train_triplets_hard_neg.jsonl"),
    )
)
# Du lieu training cuoi cung cho Bi-Encoder (tron easy va hard)
BI_ENCODER_TRAIN_MIXED_PATH = Path(
    get_env_var(
        "LAWBOT_BI_ENCODER_TRAIN_MIXED_PATH",
        str(DATA_PROCESSED_DIR / "bi_encoder_train_mixed.jsonl"),
    )
)
# Du lieu training sau khi augmentation
BI_ENCODER_TRAIN_AUGMENTED_PATH = Path(
    get_env_var(
        "LAWBOT_BI_ENCODER_TRAIN_AUGMENTED_PATH",
        str(DATA_PROCESSED_DIR / "bi_encoder_train_augmented.jsonl"),
    )
)

TRAIN_PAIRS_PATH = Path(
    get_env_var(
        "LAWBOT_TRAIN_PAIRS_PATH", str(DATA_PROCESSED_DIR / "train_pairs.jsonl")
    )
)  # Easy Negatives
TRAIN_PAIRS_HARD_NEG_PATH = Path(
    get_env_var(
        "LAWBOT_TRAIN_PAIRS_HARD_NEG_PATH",
        str(DATA_PROCESSED_DIR / "train_pairs_hard_neg.jsonl"),
    )
)  # Hard Negatives
TRAIN_PAIRS_MIXED_PATH = Path(
    get_env_var(
        "LAWBOT_TRAIN_PAIRS_MIXED_PATH",
        str(DATA_PROCESSED_DIR / "train_pairs_mixed.jsonl"),
    )
)  # Du lieu training cho Cross-Encoder (tron easy va hard)
TRAIN_PAIRS_AUGMENTED_PATH = Path(
    get_env_var(
        "LAWBOT_TRAIN_PAIRS_AUGMENTED_PATH",
        str(DATA_PROCESSED_DIR / "train_pairs_augmented.jsonl"),
    )
)  # Du lieu training cho Cross-Encoder (sau augmentation)

# --- Duong dan Mo hinh & Index ---
BI_ENCODER_PATH = Path(
    get_env_var("LAWBOT_BI_ENCODER_PATH", str(MODELS_DIR / "bi-encoder"))
)
CROSS_ENCODER_PATH = Path(
    get_env_var("LAWBOT_CROSS_ENCODER_PATH", str(MODELS_DIR / "cross-encoder"))
)
FAISS_INDEX_PATH = Path(
    get_env_var("LAWBOT_FAISS_INDEX_PATH", str(INDEXES_DIR / "legal.faiss"))
)
INDEX_TO_AID_PATH = Path(
    get_env_var("LAWBOT_INDEX_TO_AID_PATH", str(INDEXES_DIR / "index_to_aid.json"))
)

# --- Ten Model tren Hugging Face ---
BI_ENCODER_MODEL_NAME = get_env_var(
    "LAWBOT_BI_ENCODER_MODEL_NAME", "bkai-foundation-models/vietnamese-bi-encoder"
)
CROSS_ENCODER_MODEL_NAME = get_env_var(
    "LAWBOT_CROSS_ENCODER_MODEL_NAME", "vinai/phobert-large"
)

# --- PhoBERT-Law Model Path (after DAPT) ---
PHOBERT_LAW_PATH = Path(
    get_env_var("LAWBOT_PHOBERT_LAW_PATH", str(MODELS_DIR / "phobert-law"))
)

# --- Light Reranker Model Path (for Cascaded Reranking) ---
LIGHT_RERANKER_PATH = Path(
    get_env_var("LAWBOT_LIGHT_RERANKER_PATH", str(MODELS_DIR / "light-reranker"))
)
LIGHT_RERANKER_MODEL_NAME = get_env_var(
    "LAWBOT_LIGHT_RERANKER_MODEL_NAME", "vinai/phobert-base-v2"
)

# ============================================================================
# COMMON HYPERPARAMETERS (NOT PERFORMANCE MODE DEPENDENT)
# ============================================================================

# Text processing parameters
MIN_TEXT_LENGTH = get_env_int("LAWBOT_MIN_TEXT_LENGTH", 20, min_value=5, max_value=100)
MAX_TEXT_LENGTH = get_env_int(
    "LAWBOT_MAX_TEXT_LENGTH", 1000, min_value=100, max_value=5000
)
DAPT_MAX_LENGTH = get_env_int(
    "LAWBOT_DAPT_MAX_LENGTH", 128, min_value=32, max_value=512
)
DAPT_DATASET_SIZE_LIMIT = get_env_int(
    "LAWBOT_DAPT_DATASET_SIZE_LIMIT", 10000, min_value=1000, max_value=100000
)

# Data filtering parameters
EMPTY_CONTENT_THRESHOLD = get_env_float(
    "LAWBOT_EMPTY_CONTENT_THRESHOLD", 0.1, min_value=0.0, max_value=0.5
)
MIN_VALID_ARTICLES = get_env_int(
    "LAWBOT_MIN_VALID_ARTICLES", 1000, min_value=100, max_value=100000
)

# Error handling parameters
MAX_RETRIES = get_env_int("LAWBOT_MAX_RETRIES", 3, min_value=1, max_value=10)
RETRY_DELAY_SECONDS = get_env_int(
    "LAWBOT_RETRY_DELAY_SECONDS", 30, min_value=5, max_value=300
)
TIMEOUT_SECONDS = get_env_int(
    "LAWBOT_TIMEOUT_SECONDS", 300, min_value=60, max_value=3600
)

# GPU/CPU fallback parameters
FORCE_CPU_MODE = get_env_bool("LAWBOT_FORCE_CPU_MODE", False)
GPU_MEMORY_THRESHOLD_GB = get_env_float(
    "LAWBOT_GPU_MEMORY_THRESHOLD_GB", 4.0, min_value=1.0, max_value=32.0
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_config() -> bool:
    """Validate configuration settings with enhanced error handling"""
    errors = []
    warnings = []

    # Check required directories
    required_dirs = [DATA_DIR, MODELS_DIR, INDEXES_DIR, REPORTS_DIR, LOGS_DIR]
    for dir_path in required_dirs:
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                warnings.append(f"Created missing directory: {dir_path}")
            except Exception as e:
                errors.append(f"Cannot create directory {dir_path}: {e}")

    # Check required files
    required_files = [LEGAL_CORPUS_PATH, TRAIN_JSON_PATH]
    for file_path in required_files:
        if not file_path.exists():
            errors.append(f"Required file not found: {file_path}")

    # Print warnings
    if warnings:
        print("Configuration Warnings:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")

    # Raise errors if any
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(errors)
        raise ValueError(error_msg)

    return True


def print_config_summary() -> None:
    """Print comprehensive configuration summary"""
    print("=" * 80)
    print("LAWBOT BASE CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Environment: {ENVIRONMENT}")
    print(f"Debug Mode: {DEBUG}")
    print()
    print("Directories:")
    print(f"  Data: {DATA_DIR}")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Indexes: {INDEXES_DIR}")
    print(f"  Reports: {REPORTS_DIR}")
    print(f"  Logs: {LOGS_DIR}")
    print("=" * 80)


def get_config_dict() -> Dict[str, Any]:
    """Get configuration as dictionary for logging/serialization"""
    return {
        "environment": ENVIRONMENT,
        "debug": DEBUG,
        "directories": {
            "data": str(DATA_DIR),
            "models": str(MODELS_DIR),
            "indexes": str(INDEXES_DIR),
            "reports": str(REPORTS_DIR),
            "logs": str(LOGS_DIR),
        },
    }
