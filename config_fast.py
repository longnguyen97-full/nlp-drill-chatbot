#!/usr/bin/env python3
"""
FAST Mode Configuration for LawBot
=================================

This file contains configuration settings optimized for FAST mode - quick testing
and development with minimal resource usage.
"""

from config_base import *

# ============================================================================
# FAST MODE CONFIGURATION
# ============================================================================

# Performance mode identifier
PERFORMANCE_MODE = "fast"

# ============================================================================
# BI-ENCODER FAST MODE SETTINGS
# ============================================================================

BI_ENCODER_BATCH_SIZE = get_env_int("LAWBOT_BI_ENCODER_BATCH_SIZE", 8, min_value=1, max_value=128)
BI_ENCODER_EPOCHS = get_env_int("LAWBOT_BI_ENCODER_EPOCHS", 1, min_value=1, max_value=50)
BI_ENCODER_LR = get_env_float("LAWBOT_BI_ENCODER_LR", 2e-5, min_value=1e-6, max_value=1e-3)
BI_ENCODER_WARMUP_RATIO = get_env_float("LAWBOT_BI_ENCODER_WARMUP_RATIO", 0.1, min_value=0.0, max_value=0.5)
BI_ENCODER_EVAL_STEPS = get_env_int("LAWBOT_BI_ENCODER_EVAL_STEPS", 50, min_value=10, max_value=1000)
BI_ENCODER_GRADIENT_ACCUMULATION_STEPS = get_env_int("LAWBOT_BI_ENCODER_GRADIENT_ACCUMULATION_STEPS", 1, min_value=1, max_value=16)
BI_ENCODER_DATALOADER_NUM_WORKERS = get_env_int("LAWBOT_BI_ENCODER_DATALOADER_NUM_WORKERS", 2, min_value=0, max_value=16)
BI_ENCODER_DATALOADER_PIN_MEMORY = get_env_bool("LAWBOT_BI_ENCODER_DATALOADER_PIN_MEMORY", False)
BI_ENCODER_DATALOADER_PREFETCH_FACTOR = get_env_int("LAWBOT_BI_ENCODER_DATALOADER_PREFETCH_FACTOR", 1, min_value=1, max_value=4)
BI_ENCODER_EARLY_STOPPING_PATIENCE = get_env_int("LAWBOT_BI_ENCODER_EARLY_STOPPING_PATIENCE", 2, min_value=1, max_value=10)
BI_ENCODER_EARLY_STOPPING_THRESHOLD = get_env_float("LAWBOT_BI_ENCODER_EARLY_STOPPING_THRESHOLD", 0.001, min_value=0.0, max_value=0.1)

# ============================================================================
# CROSS-ENCODER FAST MODE SETTINGS
# ============================================================================

CROSS_ENCODER_BATCH_SIZE = get_env_int("LAWBOT_CROSS_ENCODER_BATCH_SIZE", 1, min_value=1, max_value=16)
CROSS_ENCODER_EPOCHS = get_env_int("LAWBOT_CROSS_ENCODER_EPOCHS", 1, min_value=1, max_value=20)
CROSS_ENCODER_LR = get_env_float("LAWBOT_CROSS_ENCODER_LR", 1e-5, min_value=1e-6, max_value=1e-3)
CROSS_ENCODER_MAX_LENGTH = get_env_int("LAWBOT_CROSS_ENCODER_MAX_LENGTH", 192, min_value=64, max_value=1024)
CROSS_ENCODER_WARMUP_RATIO = get_env_int("LAWBOT_CROSS_ENCODER_WARMUP_RATIO", 50, min_value=10, max_value=1000)
CROSS_ENCODER_EVAL_STEPS = get_env_int("LAWBOT_CROSS_ENCODER_EVAL_STEPS", 100, min_value=10, max_value=1000)
CROSS_ENCODER_GRADIENT_ACCUMULATION_STEPS = get_env_int("LAWBOT_CROSS_ENCODER_GRADIENT_ACCUMULATION_STEPS", 2, min_value=1, max_value=16)
CROSS_ENCODER_DATALOADER_NUM_WORKERS = get_env_int("LAWBOT_CROSS_ENCODER_DATALOADER_NUM_WORKERS", 1, min_value=0, max_value=8)
CROSS_ENCODER_DATALOADER_PIN_MEMORY = get_env_bool("LAWBOT_CROSS_ENCODER_DATALOADER_PIN_MEMORY", False)
CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR = get_env_int("LAWBOT_CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR", 1, min_value=1, max_value=4)
CROSS_ENCODER_EARLY_STOPPING_PATIENCE = get_env_int("LAWBOT_CROSS_ENCODER_EARLY_STOPPING_PATIENCE", 2, min_value=1, max_value=10)
CROSS_ENCODER_EARLY_STOPPING_THRESHOLD = get_env_float("LAWBOT_CROSS_ENCODER_EARLY_STOPPING_THRESHOLD", 0.001, min_value=0.0, max_value=0.1)

# ============================================================================
# LIGHT RERANKER FAST MODE SETTINGS
# ============================================================================

LIGHT_RERANKER_BATCH_SIZE = get_env_int("LAWBOT_LIGHT_RERANKER_BATCH_SIZE", 8, min_value=1, max_value=64)
LIGHT_RERANKER_EPOCHS = get_env_int("LAWBOT_LIGHT_RERANKER_EPOCHS", 1, min_value=1, max_value=20)
LIGHT_RERANKER_LR = get_env_float("LAWBOT_LIGHT_RERANKER_LR", 2e-5, min_value=1e-6, max_value=1e-3)
LIGHT_RERANKER_MAX_LENGTH = get_env_int("LAWBOT_LIGHT_RERANKER_MAX_LENGTH", 192, min_value=64, max_value=512)
LIGHT_RERANKER_WARMUP_RATIO = get_env_float("LAWBOT_LIGHT_RERANKER_WARMUP_RATIO", 0.1, min_value=0.0, max_value=0.5)
LIGHT_RERANKER_EVAL_STEPS = get_env_int("LAWBOT_LIGHT_RERANKER_EVAL_STEPS", 50, min_value=10, max_value=1000)
LIGHT_RERANKER_GRADIENT_ACCUMULATION_STEPS = get_env_int("LAWBOT_LIGHT_RERANKER_GRADIENT_ACCUMULATION_STEPS", 1, min_value=1, max_value=16)
LIGHT_RERANKER_DATALOADER_NUM_WORKERS = get_env_int("LAWBOT_LIGHT_RERANKER_DATALOADER_NUM_WORKERS", 1, min_value=0, max_value=8)
LIGHT_RERANKER_EARLY_STOPPING_PATIENCE = get_env_int("LAWBOT_LIGHT_RERANKER_EARLY_STOPPING_PATIENCE", 2, min_value=1, max_value=10)
LIGHT_RERANKER_EARLY_STOPPING_THRESHOLD = get_env_float("LAWBOT_LIGHT_RERANKER_EARLY_STOPPING_THRESHOLD", 0.001, min_value=0.0, max_value=0.1)

# ============================================================================
# PIPELINE FAST MODE SETTINGS
# ============================================================================

TOP_K_RETRIEVAL = get_env_int("LAWBOT_TOP_K_RETRIEVAL", 150, min_value=10, max_value=1000)
TOP_K_FINAL = get_env_int("LAWBOT_TOP_K_FINAL", 3, min_value=1, max_value=50)
TOP_K_LIGHT_RERANKING = get_env_int("LAWBOT_TOP_K_LIGHT_RERANKING", 40, min_value=10, max_value=200)
VALIDATION_SPLIT_RATIO = get_env_float("LAWBOT_VALIDATION_SPLIT_RATIO", 0.1, min_value=0.05, max_value=0.5)
MIN_VALIDATION_SAMPLES = get_env_int("LAWBOT_MIN_VALIDATION_SAMPLES", 50, min_value=10, max_value=10000)

# ============================================================================
# MEMORY & PERFORMANCE FAST MODE SETTINGS
# ============================================================================

FP16_TRAINING = get_env_bool("LAWBOT_FP16_TRAINING", False)
GRADIENT_CLIP_NORM = get_env_float("LAWBOT_GRADIENT_CLIP_NORM", 1.0, min_value=0.1, max_value=10.0)
MAX_MEMORY_USAGE_GB = get_env_float("LAWBOT_MAX_MEMORY_USAGE_GB", 4.0, min_value=1.0, max_value=64.0)
MEMORY_CLEANUP_THRESHOLD = get_env_float("LAWBOT_MEMORY_CLEANUP_THRESHOLD", 0.8, min_value=0.5, max_value=0.95)
BATCH_SIZE_REDUCTION_FACTOR = get_env_float("LAWBOT_BATCH_SIZE_REDUCTION_FACTOR", 0.5, min_value=0.1, max_value=0.9)

# ============================================================================
# DATA PROCESSING FAST MODE SETTINGS
# ============================================================================

AUGMENTATION_FACTOR = get_env_float("LAWBOT_AUGMENTATION_FACTOR", 1.0, min_value=1.0, max_value=5.0)
HARD_NEGATIVE_TOP_K = get_env_int("LAWBOT_HARD_NEGATIVE_TOP_K", 100, min_value=50, max_value=1000)
HARD_NEGATIVES_PER_POSITIVE = get_env_int("LAWBOT_HARD_NEGATIVES_PER_POSITIVE", 1, min_value=1, max_value=10)
NEGATIVE_SAMPLES_PER_POSITIVE = get_env_int("LAWBOT_NEGATIVE_SAMPLES_PER_POSITIVE", 2, min_value=1, max_value=20)
RANDOM_NEGATIVE_RATIO = get_env_float("LAWBOT_RANDOM_NEGATIVE_RATIO", 0.5, min_value=0.0, max_value=1.0)

# ============================================================================
# CASCADED RERANKING FAST MODE SETTINGS
# ============================================================================

LIGHT_RERANKING_WEIGHT = get_env_float("LAWBOT_LIGHT_RERANKING_WEIGHT", 0.7, min_value=0.0, max_value=1.0)
RETRIEVAL_SCORE_WEIGHT = get_env_float("LAWBOT_RETRIEVAL_SCORE_WEIGHT", 0.3, min_value=0.0, max_value=1.0)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_config_summary() -> None:
    """Print comprehensive configuration summary for FAST mode"""
    print("=" * 80)
    print("LAWBOT FAST MODE CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Environment: {ENVIRONMENT}")
    print(f"Debug Mode: {DEBUG}")
    print(f"Performance Mode: {PERFORMANCE_MODE.upper()}")
    print()
    print("Directories:")
    print(f"  Data: {DATA_DIR}")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Indexes: {INDEXES_DIR}")
    print(f"  Reports: {REPORTS_DIR}")
    print(f"  Logs: {LOGS_DIR}")
    print()
    print("FAST MODE Hyperparameters:")
    print(f"  Bi-Encoder:")
    print(f"    Batch Size: {BI_ENCODER_BATCH_SIZE}")
    print(f"    Epochs: {BI_ENCODER_EPOCHS}")
    print(f"    Learning Rate: {BI_ENCODER_LR}")
    print(f"    Warmup Ratio: {BI_ENCODER_WARMUP_RATIO}")
    print(f"    Eval Steps: {BI_ENCODER_EVAL_STEPS}")
    print(f"  Cross-Encoder:")
    print(f"    Batch Size: {CROSS_ENCODER_BATCH_SIZE}")
    print(f"    Epochs: {CROSS_ENCODER_EPOCHS}")
    print(f"    Learning Rate: {CROSS_ENCODER_LR}")
    print(f"    Max Length: {CROSS_ENCODER_MAX_LENGTH}")
    print(f"    Eval Steps: {CROSS_ENCODER_EVAL_STEPS}")
    print(f"  Light Reranker:")
    print(f"    Batch Size: {LIGHT_RERANKER_BATCH_SIZE}")
    print(f"    Epochs: {LIGHT_RERANKER_EPOCHS}")
    print(f"    Learning Rate: {LIGHT_RERANKER_LR}")
    print(f"    Max Length: {LIGHT_RERANKER_MAX_LENGTH}")
    print()
    print("FAST MODE Pipeline Settings:")
    print(f"  Top-K Retrieval: {TOP_K_RETRIEVAL}")
    print(f"  Top-K Final: {TOP_K_FINAL}")
    print(f"  Top-K Light Reranking: {TOP_K_LIGHT_RERANKING}")
    print(f"  Validation Split Ratio: {VALIDATION_SPLIT_RATIO}")
    print()
    print("FAST MODE Performance Settings:")
    print(f"  FP16 Training: {FP16_TRAINING}")
    print(f"  Gradient Clip Norm: {GRADIENT_CLIP_NORM}")
    print(f"  Max Memory Usage: {MAX_MEMORY_USAGE_GB} GB")
    print(f"  Force CPU Mode: {FORCE_CPU_MODE}")
    print()
    print("FAST MODE Data Processing:")
    print(f"  Augmentation Factor: {AUGMENTATION_FACTOR}")
    print(f"  Hard Negative Top-K: {HARD_NEGATIVE_TOP_K}")
    print(f"  Hard Negatives Per Positive: {HARD_NEGATIVES_PER_POSITIVE}")
    print(f"  Negative Samples Per Positive: {NEGATIVE_SAMPLES_PER_POSITIVE}")
    print("=" * 80)


def get_config_dict() -> Dict[str, Any]:
    """Get configuration as dictionary for logging/serialization"""
    return {
        "environment": ENVIRONMENT,
        "debug": DEBUG,
        "performance_mode": PERFORMANCE_MODE,
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
            "light_reranker": {
                "batch_size": LIGHT_RERANKER_BATCH_SIZE,
                "epochs": LIGHT_RERANKER_EPOCHS,
                "learning_rate": LIGHT_RERANKER_LR,
                "max_length": LIGHT_RERANKER_MAX_LENGTH,
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
