#!/usr/bin/env python3
"""
QUALITY Mode Configuration - Optimized for the best model performance.
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
    # --- Training Epochs for Quality Mode ---
    "BI_ENCODER_EPOCHS", "CROSS_ENCODER_EPOCHS", "LIGHT_RERANKER_EPOCHS", "DAPT_EPOCHS", "TSDAE_EPOCHS",
    # --- Batch Sizes for Quality Mode ---
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
# QUALITY MODE CONFIGURATION
# ============================================================================

PERFORMANCE_MODE = "quality"

# --- General Training Parameters (Overrides) ---
EPOCHS = get_env_int("LAWBOT_EPOCHS_QUALITY", 3)
BATCH_SIZE = get_env_int("LAWBOT_BATCH_SIZE_QUALITY", 8)
FP16_TRAINING = get_env_bool("LAWBOT_FP16_TRAINING_QUALITY", True)

# --- Model-specific Parameters ---
CROSS_ENCODER_MAX_LENGTH = 512
LIGHT_RERANKER_MAX_LENGTH = 512

# --- Training Epochs for Quality Mode ---
BI_ENCODER_EPOCHS = 2
CROSS_ENCODER_EPOCHS = 2
LIGHT_RERANKER_EPOCHS = 4
DAPT_EPOCHS = 2
TSDAE_EPOCHS = 2

# --- Batch Sizes for Quality Mode ---
BI_ENCODER_BATCH_SIZE = 32
CROSS_ENCODER_BATCH_SIZE = 16
LIGHT_RERANKER_BATCH_SIZE = 32
TSDAE_BATCH_SIZE = 32
DAPT_BATCH_SIZE = 16

# --- Pipeline & Data Parameters ---
TOP_K_RETRIEVAL = 200
TOP_K_LIGHT_RERANKING = get_env_int("LAWBOT_TOP_K_LIGHT_RERANKING_QUALITY", 80)
TOP_K_FINAL = 30
VALIDATION_SPLIT_RATIO = get_env_float("LAWBOT_VALIDATION_SPLIT_RATIO_QUALITY", 0.15)
MIN_VALIDATION_SAMPLES = get_env_int("LAWBOT_MIN_VALIDATION_SAMPLES_QUALITY", 100)
HARD_NEGATIVES_PER_POSITIVE = get_env_int("LAWBOT_HARD_NEGATIVES_PER_POSITIVE_QUALITY", 3)

# --- Evaluation K values ---
EVAL_K_VALUES = [1, 3, 5, 10, 20, 50, 100]
