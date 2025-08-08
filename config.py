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
# HYPERPARAMETERS (WITH ENHANCED VALIDATION)
# ============================================================================

# === TOI UU HOA DE KHAC PHUC OVERFITTING ===

# --- Bi-Encoder Hyperparameters (HIGHLY OPTIMIZED FOR MAXIMUM PERFORMANCE) ---
BI_ENCODER_BATCH_SIZE = get_env_int(
    "LAWBOT_BI_ENCODER_BATCH_SIZE", 32, min_value=1, max_value=128
)  # Increased for better gradient estimates
BI_ENCODER_EPOCHS = get_env_int(
    "LAWBOT_BI_ENCODER_EPOCHS", 5, min_value=1, max_value=50
)  # Increased for better learning
BI_ENCODER_LR = get_env_float(
    "LAWBOT_BI_ENCODER_LR", 2e-5, min_value=1e-6, max_value=1e-3
)  # Optimized learning rate
BI_ENCODER_WARMUP_RATIO = get_env_float(
    "LAWBOT_BI_ENCODER_WARMUP_RATIO", 0.1, min_value=0.0, max_value=0.5
)  # Use ratio for better scheduling
BI_ENCODER_EVAL_STEPS = get_env_int(
    "LAWBOT_BI_ENCODER_EVAL_STEPS", 100, min_value=10, max_value=1000
)  # Regular evaluation
BI_ENCODER_GRADIENT_ACCUMULATION_STEPS = get_env_int(
    "LAWBOT_BI_ENCODER_GRADIENT_ACCUMULATION_STEPS", 2, min_value=1, max_value=16
)  # Optimized for effective batch size
BI_ENCODER_DATALOADER_NUM_WORKERS = get_env_int(
    "LAWBOT_BI_ENCODER_DATALOADER_NUM_WORKERS", 4, min_value=0, max_value=16
)  # Better data loading

# --- Cross-Encoder Hyperparameters (OPTIMIZED FOR STABILITY & PERFORMANCE) ---
CROSS_ENCODER_BATCH_SIZE = get_env_int(
    "LAWBOT_CROSS_ENCODER_BATCH_SIZE", 2, min_value=1, max_value=16
)  # Further reduced for stability
CROSS_ENCODER_EPOCHS = get_env_int(
    "LAWBOT_CROSS_ENCODER_EPOCHS", 3, min_value=1, max_value=20
)  # Reduced for faster training
CROSS_ENCODER_LR = get_env_float(
    "LAWBOT_CROSS_ENCODER_LR", 1e-5, min_value=1e-6, max_value=1e-3
)  # Reduced for stability
CROSS_ENCODER_MAX_LENGTH = get_env_int(
    "LAWBOT_CROSS_ENCODER_MAX_LENGTH", 256, min_value=64, max_value=1024
)  # Reduced for memory efficiency
CROSS_ENCODER_WARMUP_RATIO = get_env_int(
    "LAWBOT_CROSS_ENCODER_WARMUP_RATIO", 100, min_value=10, max_value=1000
)  # Use steps instead of ratio
CROSS_ENCODER_EVAL_STEPS = get_env_int(
    "LAWBOT_CROSS_ENCODER_EVAL_STEPS", 200, min_value=10, max_value=1000
)  # Less frequent evaluation
CROSS_ENCODER_GRADIENT_ACCUMULATION_STEPS = get_env_int(
    "LAWBOT_CROSS_ENCODER_GRADIENT_ACCUMULATION_STEPS", 4, min_value=1, max_value=16
)  # Increased for effective batch size
CROSS_ENCODER_DATALOADER_NUM_WORKERS = get_env_int(
    "LAWBOT_CROSS_ENCODER_DATALOADER_NUM_WORKERS", 2, min_value=0, max_value=8
)  # Reduced for stability
CROSS_ENCODER_DATALOADER_PIN_MEMORY = get_env_bool(
    "LAWBOT_CROSS_ENCODER_DATALOADER_PIN_MEMORY", False
)  # Disabled for memory safety
CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR = get_env_int(
    "LAWBOT_CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR", 1, min_value=1, max_value=4
)  # Reduced prefetching

# === THEM CAC THAM SO MOI ===
# Early stopping parameters
BI_ENCODER_EARLY_STOPPING_PATIENCE = get_env_int(
    "LAWBOT_BI_ENCODER_EARLY_STOPPING_PATIENCE", 3, min_value=1, max_value=10
)
BI_ENCODER_EARLY_STOPPING_THRESHOLD = get_env_float(
    "LAWBOT_BI_ENCODER_EARLY_STOPPING_THRESHOLD", 0.001, min_value=0.0, max_value=0.1
)
CROSS_ENCODER_EARLY_STOPPING_PATIENCE = get_env_int(
    "LAWBOT_CROSS_ENCODER_EARLY_STOPPING_PATIENCE", 3, min_value=1, max_value=10
)
CROSS_ENCODER_EARLY_STOPPING_THRESHOLD = get_env_float(
    "LAWBOT_CROSS_ENCODER_EARLY_STOPPING_THRESHOLD", 0.001, min_value=0.0, max_value=0.1
)

# Data augmentation parameters
AUGMENTATION_FACTOR = get_env_float(
    "LAWBOT_AUGMENTATION_FACTOR", 1.5, min_value=1.0, max_value=5.0
)
LEGAL_KEYWORDS_INJECTION_RATE = get_env_float(
    "LAWBOT_LEGAL_KEYWORDS_INJECTION_RATE", 0.05, min_value=0.0, max_value=0.5
)

# Validation parameters
VALIDATION_SPLIT_RATIO = get_env_float(
    "LAWBOT_VALIDATION_SPLIT_RATIO", 0.15, min_value=0.05, max_value=0.5
)
MIN_VALIDATION_SAMPLES = get_env_int(
    "LAWBOT_MIN_VALIDATION_SAMPLES", 100, min_value=10, max_value=10000
)

# --- Tham so Pipeline & Danh gia ---
TOP_K_RETRIEVAL = get_env_int(
    "LAWBOT_TOP_K_RETRIEVAL", 100, min_value=10, max_value=1000
)
TOP_K_FINAL = get_env_int("LAWBOT_TOP_K_FINAL", 5, min_value=1, max_value=50)
EVAL_TEST_SIZE = get_env_float(
    "LAWBOT_EVAL_TEST_SIZE", 0.15, min_value=0.05, max_value=0.5
)

# --- Tham so Cascaded Reranking ---
TOP_K_LIGHT_RERANKING = get_env_int(
    "LAWBOT_TOP_K_LIGHT_RERANKING", 50, min_value=10, max_value=200
)
LIGHT_RERANKING_WEIGHT = get_env_float(
    "LAWBOT_LIGHT_RERANKING_WEIGHT", 0.7, min_value=0.0, max_value=1.0
)
RETRIEVAL_SCORE_WEIGHT = get_env_float(
    "LAWBOT_RETRIEVAL_SCORE_WEIGHT", 0.3, min_value=0.0, max_value=1.0
)
LIGHT_RERANKER_BATCH_SIZE = get_env_int(
    "LAWBOT_LIGHT_RERANKER_BATCH_SIZE", 16, min_value=1, max_value=64
)
LIGHT_RERANKER_MAX_LENGTH = get_env_int(
    "LAWBOT_LIGHT_RERANKER_MAX_LENGTH", 256, min_value=64, max_value=512
)
CROSS_ENCODER_MAX_LENGTH = get_env_int(
    "LAWBOT_CROSS_ENCODER_MAX_LENGTH", 512, min_value=128, max_value=1024
)

# --- Tham so Toi Uu Hieu Suat ---
MAX_SEQUENCE_LENGTH = get_env_int(
    "LAWBOT_MAX_SEQUENCE_LENGTH", 256, min_value=64, max_value=1024
)
GRADIENT_CLIP_NORM = get_env_float(
    "LAWBOT_GRADIENT_CLIP_NORM", 1.0, min_value=0.1, max_value=10.0
)
FP16_TRAINING = get_env_bool("LAWBOT_FP16_TRAINING", True)
WARMUP_RATIO = get_env_float("LAWBOT_WARMUP_RATIO", 0.1, min_value=0.0, max_value=0.5)

# --- DataLoader Optimization Parameters ---
BI_ENCODER_DATALOADER_NUM_WORKERS = get_env_int(
    "LAWBOT_BI_ENCODER_DATALOADER_NUM_WORKERS", 4, min_value=0, max_value=16
)
BI_ENCODER_DATALOADER_PIN_MEMORY = get_env_bool(
    "LAWBOT_BI_ENCODER_DATALOADER_PIN_MEMORY", True
)
BI_ENCODER_DATALOADER_PREFETCH_FACTOR = get_env_int(
    "LAWBOT_BI_ENCODER_DATALOADER_PREFETCH_FACTOR", 2, min_value=1, max_value=4
)
CROSS_ENCODER_DATALOADER_PIN_MEMORY = get_env_bool(
    "LAWBOT_CROSS_ENCODER_DATALOADER_PIN_MEMORY", True
)
CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR = get_env_int(
    "LAWBOT_CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR", 2, min_value=1, max_value=4
)

# ============================================================================
# MEMORY MANAGEMENT & ERROR HANDLING PARAMETERS
# ============================================================================

# Memory management parameters
MAX_MEMORY_USAGE_GB = get_env_float(
    "LAWBOT_MAX_MEMORY_USAGE_GB", 8.0, min_value=1.0, max_value=64.0
)
MEMORY_CLEANUP_THRESHOLD = get_env_float(
    "LAWBOT_MEMORY_CLEANUP_THRESHOLD", 0.8, min_value=0.5, max_value=0.95
)
BATCH_SIZE_REDUCTION_FACTOR = get_env_float(
    "LAWBOT_BATCH_SIZE_REDUCTION_FACTOR", 0.5, min_value=0.1, max_value=0.9
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
# HARD NEGATIVE SAMPLING PARAMETERS
# ============================================================================

# Hard negative sampling parameters (moved from magic numbers)
HARD_NEGATIVE_TOP_K = get_env_int(
    "LAWBOT_HARD_NEGATIVE_TOP_K", 200, min_value=50, max_value=1000
)
HARD_NEGATIVE_POSITIONS = (2, 10)  # Vị trí để lấy hard negative
HARD_NEGATIVES_PER_POSITIVE = get_env_int(
    "LAWBOT_HARD_NEGATIVES_PER_POSITIVE", 3, min_value=1, max_value=10
)

# Negative sampling parameters
NEGATIVE_SAMPLES_PER_POSITIVE = get_env_int(
    "LAWBOT_NEGATIVE_SAMPLES_PER_POSITIVE", 4, min_value=1, max_value=20
)
RANDOM_NEGATIVE_RATIO = get_env_float(
    "LAWBOT_RANDOM_NEGATIVE_RATIO", 0.5, min_value=0.0, max_value=1.0
)

# ============================================================================
# DATA PROCESSING PARAMETERS
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

# ============================================================================
# ENHANCED VALIDATION FUNCTIONS
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

    # Validate hyperparameters with enhanced checks
    hyperparameter_checks = [
        ("BI_ENCODER_BATCH_SIZE", BI_ENCODER_BATCH_SIZE, 1, 128),
        ("CROSS_ENCODER_BATCH_SIZE", CROSS_ENCODER_BATCH_SIZE, 1, 16),
        ("BI_ENCODER_LR", BI_ENCODER_LR, 1e-6, 1e-3),
        ("CROSS_ENCODER_LR", CROSS_ENCODER_LR, 1e-6, 1e-3),
        ("TOP_K_RETRIEVAL", TOP_K_RETRIEVAL, 10, 1000),
        ("TOP_K_FINAL", TOP_K_FINAL, 1, 50),
    ]

    for name, value, min_val, max_val in hyperparameter_checks:
        if value < min_val or value > max_val:
            errors.append(f"{name} ({value}) must be between {min_val} and {max_val}")

    # Validate weight combinations
    if abs(LIGHT_RERANKING_WEIGHT + RETRIEVAL_SCORE_WEIGHT - 1.0) > 0.01:
        warnings.append(
            "LIGHT_RERANKING_WEIGHT + RETRIEVAL_SCORE_WEIGHT should sum to 1.0"
        )

    # Validate memory settings
    if MAX_MEMORY_USAGE_GB < 1.0:
        warnings.append("MAX_MEMORY_USAGE_GB is very low, may cause issues")

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
    print("LAWBOT CONFIGURATION SUMMARY")
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
    print()
    print("Hyperparameters:")
    print(f"  Bi-Encoder:")
    print(f"    Batch Size: {BI_ENCODER_BATCH_SIZE}")
    print(f"    Epochs: {BI_ENCODER_EPOCHS}")
    print(f"    Learning Rate: {BI_ENCODER_LR}")
    print(f"    Warmup Ratio: {BI_ENCODER_WARMUP_RATIO}")
    print(f"  Cross-Encoder:")
    print(f"    Batch Size: {CROSS_ENCODER_BATCH_SIZE}")
    print(f"    Epochs: {CROSS_ENCODER_EPOCHS}")
    print(f"    Learning Rate: {CROSS_ENCODER_LR}")
    print(f"    Max Length: {CROSS_ENCODER_MAX_LENGTH}")
    print()
    print("Pipeline Settings:")
    print(f"  Top-K Retrieval: {TOP_K_RETRIEVAL}")
    print(f"  Top-K Final: {TOP_K_FINAL}")
    print(f"  Validation Split Ratio: {VALIDATION_SPLIT_RATIO}")
    print()
    print("Performance Settings:")
    print(f"  FP16 Training: {FP16_TRAINING}")
    print(f"  Gradient Clip Norm: {GRADIENT_CLIP_NORM}")
    print(f"  Max Memory Usage: {MAX_MEMORY_USAGE_GB} GB")
    print(f"  Force CPU Mode: {FORCE_CPU_MODE}")
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
        "hyperparameters": {
            "bi_encoder": {
                "batch_size": BI_ENCODER_BATCH_SIZE,
                "epochs": BI_ENCODER_EPOCHS,
                "learning_rate": BI_ENCODER_LR,
                "warmup_ratio": BI_ENCODER_WARMUP_RATIO,
            },
            "cross_encoder": {
                "batch_size": CROSS_ENCODER_BATCH_SIZE,
                "epochs": CROSS_ENCODER_EPOCHS,
                "learning_rate": CROSS_ENCODER_LR,
                "max_length": CROSS_ENCODER_MAX_LENGTH,
            },
        },
        "pipeline": {
            "top_k_retrieval": TOP_K_RETRIEVAL,
            "top_k_final": TOP_K_FINAL,
            "validation_split_ratio": VALIDATION_SPLIT_RATIO,
        },
        "performance": {
            "fp16_training": FP16_TRAINING,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "max_memory_usage_gb": MAX_MEMORY_USAGE_GB,
            "force_cpu_mode": FORCE_CPU_MODE,
        },
    }


# Auto-validate on import
if __name__ != "__main__":
    try:
        validate_config()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please check your environment variables and file paths.")
        sys.exit(1)
