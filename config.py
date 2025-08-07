import os
from pathlib import Path
from typing import Optional

# ============================================================================
# ENVIRONMENT VARIABLES SUPPORT
# ============================================================================


def get_env_var(key: str, default: str = None) -> Optional[str]:
    """Get environment variable with fallback to default"""
    return os.getenv(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable"""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def get_env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable"""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float environment variable"""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


# ============================================================================
# BASE CONFIGURATION
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
# HYPERPARAMETERS (WITH ENVIRONMENT SUPPORT)
# ============================================================================

# === TOI UU HOA DE KHAC PHUC OVERFITTING ===

# --- Bi-Encoder Hyperparameters (HIGHLY OPTIMIZED FOR MAXIMUM PERFORMANCE) ---
BI_ENCODER_BATCH_SIZE = get_env_int(
    "LAWBOT_BI_ENCODER_BATCH_SIZE", 32
)  # Increased for better gradient estimates
BI_ENCODER_EPOCHS = get_env_int(
    "LAWBOT_BI_ENCODER_EPOCHS", 5
)  # Increased for better learning
BI_ENCODER_LR = get_env_float("LAWBOT_BI_ENCODER_LR", 2e-5)  # Optimized learning rate
BI_ENCODER_WARMUP_RATIO = get_env_float(
    "LAWBOT_BI_ENCODER_WARMUP_RATIO", 0.1
)  # Use ratio for better scheduling
BI_ENCODER_EVAL_STEPS = get_env_int(
    "LAWBOT_BI_ENCODER_EVAL_STEPS", 100
)  # Regular evaluation
BI_ENCODER_GRADIENT_ACCUMULATION_STEPS = get_env_int(
    "LAWBOT_BI_ENCODER_GRADIENT_ACCUMULATION_STEPS", 2
)  # Optimized for effective batch size
BI_ENCODER_DATALOADER_NUM_WORKERS = get_env_int(
    "LAWBOT_BI_ENCODER_DATALOADER_NUM_WORKERS", 4
)  # Better data loading

# --- Cross-Encoder Hyperparameters (HIGHLY OPTIMIZED FOR MAXIMUM PERFORMANCE) ---
CROSS_ENCODER_BATCH_SIZE = get_env_int(
    "LAWBOT_CROSS_ENCODER_BATCH_SIZE", 16
)  # Increased for better gradient estimates
CROSS_ENCODER_EPOCHS = get_env_int(
    "LAWBOT_CROSS_ENCODER_EPOCHS", 8
)  # Increased for better learning
CROSS_ENCODER_LR = get_env_float(
    "LAWBOT_CROSS_ENCODER_LR", 2e-5
)  # Optimized learning rate
CROSS_ENCODER_MAX_LENGTH = get_env_int(
    "LAWBOT_CROSS_ENCODER_MAX_LENGTH", 512
)  # Increased for better context
CROSS_ENCODER_WARMUP_RATIO = get_env_float(
    "LAWBOT_CROSS_ENCODER_WARMUP_RATIO", 0.1
)  # Use ratio for better scheduling
CROSS_ENCODER_EVAL_STEPS = get_env_int(
    "LAWBOT_CROSS_ENCODER_EVAL_STEPS", 100
)  # Regular evaluation
CROSS_ENCODER_GRADIENT_ACCUMULATION_STEPS = get_env_int(
    "LAWBOT_CROSS_ENCODER_GRADIENT_ACCUMULATION_STEPS", 2
)  # Optimized for effective batch size
CROSS_ENCODER_DATALOADER_NUM_WORKERS = get_env_int(
    "LAWBOT_CROSS_ENCODER_DATALOADER_NUM_WORKERS", 4
)  # Better data loading
CROSS_ENCODER_DATALOADER_PIN_MEMORY = get_env_bool(
    "LAWBOT_CROSS_ENCODER_DATALOADER_PIN_MEMORY", True
)  # Memory optimization
CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR = get_env_int(
    "LAWBOT_CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR", 2
)  # Data prefetching

# === THEM CAC THAM SO MOI ===
# Early stopping parameters
BI_ENCODER_EARLY_STOPPING_PATIENCE = get_env_int(
    "LAWBOT_BI_ENCODER_EARLY_STOPPING_PATIENCE", 3
)
BI_ENCODER_EARLY_STOPPING_THRESHOLD = get_env_float(
    "LAWBOT_BI_ENCODER_EARLY_STOPPING_THRESHOLD", 0.001
)
CROSS_ENCODER_EARLY_STOPPING_PATIENCE = get_env_int(
    "LAWBOT_CROSS_ENCODER_EARLY_STOPPING_PATIENCE", 3
)
CROSS_ENCODER_EARLY_STOPPING_THRESHOLD = get_env_float(
    "LAWBOT_CROSS_ENCODER_EARLY_STOPPING_THRESHOLD", 0.001
)

# Data augmentation parameters
AUGMENTATION_FACTOR = get_env_float("LAWBOT_AUGMENTATION_FACTOR", 1.5)
LEGAL_KEYWORDS_INJECTION_RATE = get_env_float(
    "LAWBOT_LEGAL_KEYWORDS_INJECTION_RATE", 0.05
)

# Validation parameters
VALIDATION_SPLIT_RATIO = get_env_float("LAWBOT_VALIDATION_SPLIT_RATIO", 0.15)
MIN_VALIDATION_SAMPLES = get_env_int("LAWBOT_MIN_VALIDATION_SAMPLES", 100)

# --- Tham so Pipeline & Danh gia ---
TOP_K_RETRIEVAL = get_env_int("LAWBOT_TOP_K_RETRIEVAL", 100)
TOP_K_FINAL = get_env_int("LAWBOT_TOP_K_FINAL", 5)
EVAL_TEST_SIZE = get_env_float("LAWBOT_EVAL_TEST_SIZE", 0.15)

# --- Tham so Cascaded Reranking ---
TOP_K_LIGHT_RERANKING = get_env_int("LAWBOT_TOP_K_LIGHT_RERANKING", 50)
LIGHT_RERANKING_WEIGHT = get_env_float("LAWBOT_LIGHT_RERANKING_WEIGHT", 0.7)
RETRIEVAL_SCORE_WEIGHT = get_env_float("LAWBOT_RETRIEVAL_SCORE_WEIGHT", 0.3)
LIGHT_RERANKER_BATCH_SIZE = get_env_int("LAWBOT_LIGHT_RERANKER_BATCH_SIZE", 16)
LIGHT_RERANKER_MAX_LENGTH = get_env_int("LAWBOT_LIGHT_RERANKER_MAX_LENGTH", 256)
CROSS_ENCODER_MAX_LENGTH = get_env_int("LAWBOT_CROSS_ENCODER_MAX_LENGTH", 512)

# --- Tham so Toi Uu Hieu Suat ---
MAX_SEQUENCE_LENGTH = get_env_int("LAWBOT_MAX_SEQUENCE_LENGTH", 256)
GRADIENT_CLIP_NORM = get_env_float("LAWBOT_GRADIENT_CLIP_NORM", 1.0)
FP16_TRAINING = get_env_bool("LAWBOT_FP16_TRAINING", True)
WARMUP_RATIO = get_env_float("LAWBOT_WARMUP_RATIO", 0.1)

# --- DataLoader Optimization Parameters ---
BI_ENCODER_DATALOADER_NUM_WORKERS = get_env_int(
    "LAWBOT_BI_ENCODER_DATALOADER_NUM_WORKERS", 4
)
BI_ENCODER_DATALOADER_PIN_MEMORY = get_env_bool(
    "LAWBOT_BI_ENCODER_DATALOADER_PIN_MEMORY", True
)
BI_ENCODER_DATALOADER_PREFETCH_FACTOR = get_env_int(
    "LAWBOT_BI_ENCODER_DATALOADER_PREFETCH_FACTOR", 2
)
CROSS_ENCODER_DATALOADER_PIN_MEMORY = get_env_bool(
    "LAWBOT_CROSS_ENCODER_DATALOADER_PIN_MEMORY", True
)
CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR = get_env_int(
    "LAWBOT_CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR", 2
)

# ============================================================================
# HARD NEGATIVE SAMPLING PARAMETERS
# ============================================================================

# Hard negative sampling parameters (moved from magic numbers)
HARD_NEGATIVE_TOP_K = get_env_int("LAWBOT_HARD_NEGATIVE_TOP_K", 200)
HARD_NEGATIVE_POSITIONS = (2, 10)  # Vị trí để lấy hard negative
HARD_NEGATIVES_PER_POSITIVE = get_env_int("LAWBOT_HARD_NEGATIVES_PER_POSITIVE", 3)

# Negative sampling parameters
NEGATIVE_SAMPLES_PER_POSITIVE = get_env_int("LAWBOT_NEGATIVE_SAMPLES_PER_POSITIVE", 4)
RANDOM_NEGATIVE_RATIO = get_env_float("LAWBOT_RANDOM_NEGATIVE_RATIO", 0.5)

# ============================================================================
# DATA PROCESSING PARAMETERS
# ============================================================================

# Text processing parameters
MIN_TEXT_LENGTH = get_env_int("LAWBOT_MIN_TEXT_LENGTH", 20)
MAX_TEXT_LENGTH = get_env_int("LAWBOT_MAX_TEXT_LENGTH", 1000)
DAPT_MAX_LENGTH = get_env_int("LAWBOT_DAPT_MAX_LENGTH", 128)
DAPT_DATASET_SIZE_LIMIT = get_env_int("LAWBOT_DAPT_DATASET_SIZE_LIMIT", 10000)

# Data filtering parameters
EMPTY_CONTENT_THRESHOLD = get_env_float("LAWBOT_EMPTY_CONTENT_THRESHOLD", 0.1)
MIN_VALID_ARTICLES = get_env_int("LAWBOT_MIN_VALID_ARTICLES", 1000)

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def validate_config():
    """Validate configuration settings"""
    errors = []

    # Check required directories
    required_dirs = [DATA_DIR, MODELS_DIR, INDEXES_DIR, REPORTS_DIR, LOGS_DIR]
    for dir_path in required_dirs:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)

    # Check required files
    required_files = [LEGAL_CORPUS_PATH, TRAIN_JSON_PATH]
    for file_path in required_files:
        if not file_path.exists():
            errors.append(f"Required file not found: {file_path}")

    # Validate hyperparameters
    if BI_ENCODER_BATCH_SIZE <= 0:
        errors.append("BI_ENCODER_BATCH_SIZE must be positive")
    if CROSS_ENCODER_BATCH_SIZE <= 0:
        errors.append("CROSS_ENCODER_BATCH_SIZE must be positive")
    if BI_ENCODER_LR <= 0:
        errors.append("BI_ENCODER_LR must be positive")
    if CROSS_ENCODER_LR <= 0:
        errors.append("CROSS_ENCODER_LR must be positive")

    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))

    return True


def print_config_summary():
    """Print configuration summary"""
    print("=" * 60)
    print("LAWBOT CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Environment: {ENVIRONMENT}")
    print(f"Debug Mode: {DEBUG}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Indexes Directory: {INDEXES_DIR}")
    print(f"Reports Directory: {REPORTS_DIR}")
    print(f"Logs Directory: {LOGS_DIR}")
    print()
    print("Hyperparameters:")
    print(f"  Bi-Encoder Batch Size: {BI_ENCODER_BATCH_SIZE}")
    print(f"  Bi-Encoder Epochs: {BI_ENCODER_EPOCHS}")
    print(f"  Bi-Encoder Learning Rate: {BI_ENCODER_LR}")
    print(f"  Cross-Encoder Batch Size: {CROSS_ENCODER_BATCH_SIZE}")
    print(f"  Cross-Encoder Epochs: {CROSS_ENCODER_EPOCHS}")
    print(f"  Cross-Encoder Learning Rate: {CROSS_ENCODER_LR}")
    print()
    print("Pipeline Settings:")
    print(f"  Top-K Retrieval: {TOP_K_RETRIEVAL}")
    print(f"  Top-K Final: {TOP_K_FINAL}")
    print(f"  Validation Split Ratio: {VALIDATION_SPLIT_RATIO}")
    print("=" * 60)


# Auto-validate on import
if __name__ != "__main__":
    try:
        validate_config()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please check your environment variables and file paths.")
