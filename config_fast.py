#!/usr/bin/env python3
"""
FAST Mode Configuration - Optimized for quick development and testing.
"""

from config_base import *
from config_base import __all__ as config_base_all

__all__ = [
    "PERFORMANCE_MODE",
    # General Training Params (epochs and batch sizes are mode-specific)
    "EPOCHS", "BATCH_SIZE", "FP16_TRAINING",
    # Model-specific Params
    "CROSS_ENCODER_MAX_LENGTH", "LIGHT_RERANKER_MAX_LENGTH",
    # Pipeline Params
    "TOP_K_RETRIEVAL", "TOP_K_FINAL", "TOP_K_LIGHT_RERANKING",
    "VALIDATION_SPLIT_RATIO", "MIN_VALIDATION_SAMPLES",
    # Data Processing
    "HARD_NEGATIVES_PER_POSITIVE",
    # Evaluation
    "EVAL_K_VALUES",
    # --- Training Epochs for Fast Mode ---
    "BI_ENCODER_EPOCHS", "CROSS_ENCODER_EPOCHS", "LIGHT_RERANKER_EPOCHS", "DAPT_EPOCHS", "TSDAE_EPOCHS",
    # --- Batch Sizes for Fast Mode ---
    "BI_ENCODER_BATCH_SIZE", "CROSS_ENCODER_BATCH_SIZE", "LIGHT_RERANKER_BATCH_SIZE", "TSDAE_BATCH_SIZE", "DAPT_BATCH_SIZE"
]
__all__ += config_base_all  # Inherit all from base
__all__ += [
    # DataLoader Params
    "BI_ENCODER_DATALOADER_NUM_WORKERS", "BI_ENCODER_DATALOADER_PIN_MEMORY", "BI_ENCODER_DATALOADER_PREFETCH_FACTOR",
    "CROSS_ENCODER_DATALOADER_NUM_WORKERS", "CROSS_ENCODER_DATALOADER_PIN_MEMORY", "CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR",
    "LIGHT_RERANKER_DATALOADER_NUM_WORKERS"
]

# ============================================================================
# FAST MODE CONFIGURATION
# ============================================================================

PERFORMANCE_MODE = "fast"

# --- General Training Parameters (Overrides) ---
EPOCHS = get_env_int("LAWBOT_EPOCHS_FAST", 1)
BATCH_SIZE = get_env_int("LAWBOT_BATCH_SIZE_FAST", 4)
FP16_TRAINING = get_env_bool("LAWBOT_FP16_TRAINING_FAST", False)

# --- Model-specific Parameters ---
CROSS_ENCODER_MAX_LENGTH = 512
LIGHT_RERANKER_MAX_LENGTH = 256

# --- Training Epochs for Fast Mode ---
BI_ENCODER_EPOCHS = 1
CROSS_ENCODER_EPOCHS = 1
LIGHT_RERANKER_EPOCHS = 2
DAPT_EPOCHS = 1
TSDAE_EPOCHS = 1

# --- Batch Sizes for Fast Mode ---
BI_ENCODER_BATCH_SIZE = 16
CROSS_ENCODER_BATCH_SIZE = 8
LIGHT_RERANKER_BATCH_SIZE = 16
TSDAE_BATCH_SIZE = 16
DAPT_BATCH_SIZE = 8

# --- Pipeline & Data Parameters ---
TOP_K_RETRIEVAL = 200
TOP_K_LIGHT_RERANKING = get_env_int("LAWBOT_TOP_K_LIGHT_RERANKING_FAST", 80)
TOP_K_FINAL = 20
VALIDATION_SPLIT_RATIO = get_env_float("LAWBOT_VALIDATION_SPLIT_RATIO_FAST", 0.1)
MIN_VALIDATION_SAMPLES = get_env_int("LAWBOT_MIN_VALIDATION_SAMPLES_FAST", 50)
HARD_NEGATIVES_PER_POSITIVE = get_env_int("LAWBOT_HARD_NEGATIVES_PER_POSITIVE_FAST", 1)

# --- Evaluation K values ---
EVAL_K_VALUES = [1, 3, 5, 10, 20, 50]
