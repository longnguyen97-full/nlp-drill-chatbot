#!/usr/bin/env python3
"""
Main Configuration for LawBot
============================

This file dynamically imports configuration settings based on the performance mode.
It supports FAST and QUALITY modes with separate configuration files.
"""

import os
import sys
from pathlib import Path

# ============================================================================
# PERFORMANCE MODE DETECTION
# ============================================================================

def get_performance_mode() -> str:
    """Get the current performance mode from environment variable"""
    mode = os.getenv("LAWBOT_PERFORMANCE_MODE", "quality").lower()
    if mode not in ["fast", "quality"]:
        print(f"⚠️  Invalid PERFORMANCE_MODE '{mode}', using 'quality'")
        mode = "quality"
    return mode

# ============================================================================
# DYNAMIC CONFIG IMPORT
# ============================================================================

def load_config():
    """Dynamically load configuration based on performance mode"""
    mode = get_performance_mode()
    
    if mode == "fast":
        print(f"Loading FAST mode configuration...")
        import config_fast
        print(f"FAST mode configuration loaded successfully!")
        return mode, config_fast
    else:
        print(f"Loading QUALITY mode configuration...")
        import config_quality
        print(f"QUALITY mode configuration loaded successfully!")
        return mode, config_quality

# Load the appropriate configuration
PERFORMANCE_MODE, config_module = load_config()

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_config() -> bool:
    """Validate configuration settings with enhanced error handling"""
    try:
        from config_base import validate_config as base_validate
        return base_validate()
    except ImportError:
        print("⚠️  Could not import base config validation")
        return True

def print_config_summary() -> None:
    """Print comprehensive configuration summary"""
    if PERFORMANCE_MODE == "fast":
        from config_fast import print_config_summary as fast_print
        fast_print()
    else:
        from config_quality import print_config_summary as quality_print
        quality_print()

def get_config_dict():
    """Get configuration as dictionary for logging/serialization"""
    if PERFORMANCE_MODE == "fast":
        from config_fast import get_config_dict as fast_dict
        return fast_dict()
    else:
        from config_quality import get_config_dict as quality_dict
        return quality_dict()

# ============================================================================
# CONFIGURATION EXPORTS
# ============================================================================

# Export all configuration variables from the loaded config
if PERFORMANCE_MODE == "fast":
    from config_fast import (
        BI_ENCODER_BATCH_SIZE, BI_ENCODER_EPOCHS, BI_ENCODER_LR, BI_ENCODER_WARMUP_RATIO,
        BI_ENCODER_EVAL_STEPS, BI_ENCODER_GRADIENT_ACCUMULATION_STEPS,
        BI_ENCODER_DATALOADER_NUM_WORKERS, BI_ENCODER_DATALOADER_PIN_MEMORY,
        BI_ENCODER_DATALOADER_PREFETCH_FACTOR, BI_ENCODER_EARLY_STOPPING_PATIENCE,
        BI_ENCODER_EARLY_STOPPING_THRESHOLD, CROSS_ENCODER_BATCH_SIZE,
        CROSS_ENCODER_EPOCHS, CROSS_ENCODER_LR, CROSS_ENCODER_MAX_LENGTH,
        CROSS_ENCODER_WARMUP_RATIO, CROSS_ENCODER_EVAL_STEPS,
        CROSS_ENCODER_GRADIENT_ACCUMULATION_STEPS, CROSS_ENCODER_DATALOADER_NUM_WORKERS,
        CROSS_ENCODER_DATALOADER_PIN_MEMORY, CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR,
        CROSS_ENCODER_EARLY_STOPPING_PATIENCE, CROSS_ENCODER_EARLY_STOPPING_THRESHOLD,
        LIGHT_RERANKER_BATCH_SIZE, LIGHT_RERANKER_EPOCHS, LIGHT_RERANKER_LR,
        LIGHT_RERANKER_MAX_LENGTH, LIGHT_RERANKER_WARMUP_RATIO, LIGHT_RERANKER_EVAL_STEPS,
        LIGHT_RERANKER_GRADIENT_ACCUMULATION_STEPS, LIGHT_RERANKER_DATALOADER_NUM_WORKERS,
        LIGHT_RERANKER_EARLY_STOPPING_PATIENCE, LIGHT_RERANKER_EARLY_STOPPING_THRESHOLD,
        TOP_K_RETRIEVAL, TOP_K_FINAL, TOP_K_LIGHT_RERANKING, VALIDATION_SPLIT_RATIO,
        MIN_VALIDATION_SAMPLES, FP16_TRAINING, GRADIENT_CLIP_NORM, MAX_MEMORY_USAGE_GB,
        MEMORY_CLEANUP_THRESHOLD, BATCH_SIZE_REDUCTION_FACTOR, AUGMENTATION_FACTOR,
        HARD_NEGATIVE_TOP_K, HARD_NEGATIVES_PER_POSITIVE, NEGATIVE_SAMPLES_PER_POSITIVE,
        RANDOM_NEGATIVE_RATIO, LIGHT_RERANKING_WEIGHT, RETRIEVAL_SCORE_WEIGHT
    )
else:
    from config_quality import (
        BI_ENCODER_BATCH_SIZE, BI_ENCODER_EPOCHS, BI_ENCODER_LR, BI_ENCODER_WARMUP_RATIO,
        BI_ENCODER_EVAL_STEPS, BI_ENCODER_GRADIENT_ACCUMULATION_STEPS,
        BI_ENCODER_DATALOADER_NUM_WORKERS, BI_ENCODER_DATALOADER_PIN_MEMORY,
        BI_ENCODER_DATALOADER_PREFETCH_FACTOR, BI_ENCODER_EARLY_STOPPING_PATIENCE,
        BI_ENCODER_EARLY_STOPPING_THRESHOLD, CROSS_ENCODER_BATCH_SIZE,
        CROSS_ENCODER_EPOCHS, CROSS_ENCODER_LR, CROSS_ENCODER_MAX_LENGTH,
        CROSS_ENCODER_WARMUP_RATIO, CROSS_ENCODER_EVAL_STEPS,
        CROSS_ENCODER_GRADIENT_ACCUMULATION_STEPS, CROSS_ENCODER_DATALOADER_NUM_WORKERS,
        CROSS_ENCODER_DATALOADER_PIN_MEMORY, CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR,
        CROSS_ENCODER_EARLY_STOPPING_PATIENCE, CROSS_ENCODER_EARLY_STOPPING_THRESHOLD,
        LIGHT_RERANKER_BATCH_SIZE, LIGHT_RERANKER_EPOCHS, LIGHT_RERANKER_LR,
        LIGHT_RERANKER_MAX_LENGTH, LIGHT_RERANKER_WARMUP_RATIO, LIGHT_RERANKER_EVAL_STEPS,
        LIGHT_RERANKER_GRADIENT_ACCUMULATION_STEPS, LIGHT_RERANKER_DATALOADER_NUM_WORKERS,
        LIGHT_RERANKER_EARLY_STOPPING_PATIENCE, LIGHT_RERANKER_EARLY_STOPPING_THRESHOLD,
        TOP_K_RETRIEVAL, TOP_K_FINAL, TOP_K_LIGHT_RERANKING, VALIDATION_SPLIT_RATIO,
        MIN_VALIDATION_SAMPLES, FP16_TRAINING, GRADIENT_CLIP_NORM, MAX_MEMORY_USAGE_GB,
        MEMORY_CLEANUP_THRESHOLD, BATCH_SIZE_REDUCTION_FACTOR, AUGMENTATION_FACTOR,
        HARD_NEGATIVE_TOP_K, HARD_NEGATIVES_PER_POSITIVE, NEGATIVE_SAMPLES_PER_POSITIVE,
        RANDOM_NEGATIVE_RATIO, LIGHT_RERANKING_WEIGHT, RETRIEVAL_SCORE_WEIGHT
    )

# Also import base config variables
from config_base import (
    ROOT_DIR, ENVIRONMENT, DEBUG, DATA_DIR, MODELS_DIR, INDEXES_DIR, REPORTS_DIR, LOGS_DIR,
    DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_VALIDATION_DIR, LEGAL_CORPUS_PATH, TRAIN_JSON_PATH,
    PUBLIC_TEST_JSON_PATH, TRAIN_EXTENDED_PATH, TRAIN_ENHANCED_AUGMENTED_PATH,
    TRAIN_EXTENDED_ENHANCED_AUGMENTED_PATH, TRAIN_SPLIT_JSON_PATH, VAL_SPLIT_JSON_PATH,
    AID_MAP_PATH, DOC_ID_TO_AIDS_PATH, TRAIN_TRIPLETS_EASY_PATH,
    TRAIN_TRIPLETS_EASY_FOR_TRAINING_PATH, BI_ENCODER_VALIDATION_PATH,
    TRAIN_TRIPLETS_HARD_NEG_PATH, BI_ENCODER_TRAIN_MIXED_PATH, BI_ENCODER_TRAIN_AUGMENTED_PATH,
    TRAIN_PAIRS_PATH, TRAIN_PAIRS_HARD_NEG_PATH, TRAIN_PAIRS_MIXED_PATH,
    TRAIN_PAIRS_AUGMENTED_PATH, BI_ENCODER_PATH, CROSS_ENCODER_PATH, FAISS_INDEX_PATH,
    INDEX_TO_AID_PATH, BI_ENCODER_MODEL_NAME, CROSS_ENCODER_MODEL_NAME, PHOBERT_LAW_PATH,
    LIGHT_RERANKER_PATH, LIGHT_RERANKER_MODEL_NAME, MIN_TEXT_LENGTH, MAX_TEXT_LENGTH,
    DAPT_MAX_LENGTH, DAPT_DATASET_SIZE_LIMIT, EMPTY_CONTENT_THRESHOLD, MIN_VALID_ARTICLES,
    MAX_RETRIES, RETRY_DELAY_SECONDS, TIMEOUT_SECONDS, FORCE_CPU_MODE, GPU_MEMORY_THRESHOLD_GB
)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print_config_summary()
