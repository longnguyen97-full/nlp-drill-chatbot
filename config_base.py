#!/usr/bin/env python3
"""
Base Configuration for LawBot
============================

Contains shared configuration settings, paths, and utility functions
for all performance modes.
"""

import os
from pathlib import Path
from typing import Optional
import logging

__all__ = [
    # Environment
    "ENVIRONMENT", "DEBUG",
    # Core Directories
    "ROOT_DIR", "DATA_DIR", "MODELS_DIR", "INDEXES_DIR", "REPORTS_DIR", "LOGS_DIR",
    "DATA_RAW_DIR", "DATA_PROCESSED_DIR", "DATA_VALIDATION_DIR",
    # Key File Paths
    "LEGAL_CORPUS_PATH", "TRAIN_JSON_PATH", "PUBLIC_TEST_JSON_PATH",
    "TRAIN_SPLIT_JSON_PATH", "VAL_SPLIT_JSON_PATH", "AID_MAP_PATH",
    "BI_ENCODER_TRAIN_MIXED_PATH", "CROSS_ENCODER_TRAIN_PATH",
    # Model & Index Paths
    "BI_ENCODER_PATH", "CROSS_ENCODER_PATH", "LIGHT_RERANKER_PATH",
    "DAPT_ADAPTED_MODEL_PATH", "TSDAE_ADAPTED_MODEL_PATH",
    "FAISS_INDEX_PATH", "INDEX_TO_AID_PATH", "PHOBERT_LAW_PATH",
    # Base Model Names
    "BI_ENCODER_MODEL_NAME", "CROSS_ENCODER_MODEL_NAME", "LIGHT_RERANKER_MODEL_NAME",
    # Domain Adaptation Config
    "DAPT_MAX_LENGTH", "DAPT_DATASET_SIZE_LIMIT", "TSDAE_EPOCHS", "TSDAE_BATCH_SIZE",
    # System & Error Handling
    "MAX_RETRIES", "RETRY_DELAY_SECONDS", "FORCE_CPU_MODE",
    # Env Var Getters
    "get_env_var", "get_env_bool", "get_env_int", "get_env_float",
    # DataLoader Params
    "BI_ENCODER_DATALOADER_NUM_WORKERS", "BI_ENCODER_DATALOADER_PIN_MEMORY", "BI_ENCODER_DATALOADER_PREFETCH_FACTOR",
    "CROSS_ENCODER_DATALOADER_NUM_WORKERS", "CROSS_ENCODER_DATALOADER_PIN_MEMORY", "CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR",
    "LIGHT_RERANKER_DATALOADER_NUM_WORKERS",
    # Bi-Encoder Training Params
    "BI_ENCODER_LR", "BI_ENCODER_WARMUP_RATIO", "BI_ENCODER_EVAL_STEPS",
    "BI_ENCODER_EARLY_STOPPING_PATIENCE", "BI_ENCODER_EARLY_STOPPING_THRESHOLD", "BI_ENCODER_GRADIENT_ACCUMULATION_STEPS",
    # Cross-Encoder Training Params
    "CROSS_ENCODER_LR", "CROSS_ENCODER_WARMUP_RATIO", "CROSS_ENCODER_EVAL_STEPS",
    "CROSS_ENCODER_EARLY_STOPPING_PATIENCE", "CROSS_ENCODER_EARLY_STOPPING_THRESHOLD", "CROSS_ENCODER_GRADIENT_ACCUMULATION_STEPS",
    # Light-Reranker Training Params
    "LIGHT_RERANKER_LR", "LIGHT_RERANKER_WARMUP_RATIO", "LIGHT_RERANKER_EVAL_STEPS",
    "LIGHT_RERANKER_EARLY_STOPPING_PATIENCE", "LIGHT_RERANKER_EARLY_STOPPING_THRESHOLD", "LIGHT_RERANKER_GRADIENT_ACCUMULATION_STEPS"
]

# ============================================================================
# ENVIRONMENT VARIABLE UTILITIES
# ============================================================================

def get_env_var(key: str, default: str = None) -> Optional[str]:
    """Get environment variable with a fallback default."""
    return os.getenv(key, default)

def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable with validation."""
    value = str(os.getenv(key, str(default))).lower()
    return value in ("true", "1", "yes", "on")

def get_env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default

def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default

# ============================================================================
# CORE PATHS & ENVIRONMENT
# ============================================================================

ROOT_DIR = Path(__file__).parent
ENVIRONMENT = get_env_var("LAWBOT_ENV", "development")
DEBUG = get_env_bool("LAWBOT_DEBUG", True)

# --- Core Directories ---
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
INDEXES_DIR = ROOT_DIR / "indexes"
REPORTS_DIR = ROOT_DIR / "reports"
LOGS_DIR = ROOT_DIR / "logs"

# --- Data Subdirectories ---
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_VALIDATION_DIR = DATA_DIR / "validation"

# ============================================================================
# KEY FILE PATHS
# ============================================================================

# --- Raw Data ---
LEGAL_CORPUS_PATH = DATA_RAW_DIR / "legal_corpus.json"
TRAIN_JSON_PATH = DATA_RAW_DIR / "train.json"
PUBLIC_TEST_JSON_PATH = DATA_RAW_DIR / "public_test.json"

# --- Processed Data ---
TRAIN_SPLIT_JSON_PATH = DATA_PROCESSED_DIR / "train_split.json"
VAL_SPLIT_JSON_PATH = DATA_PROCESSED_DIR / "validation_split.json"
AID_MAP_PATH = DATA_PROCESSED_DIR / "aid_map.pkl"
BI_ENCODER_TRAIN_MIXED_PATH = DATA_PROCESSED_DIR / "bi_encoder_train_mixed.jsonl"
CROSS_ENCODER_TRAIN_PATH = DATA_PROCESSED_DIR / "cross_encoder_train.jsonl"

# ============================================================================
# MODEL & INDEX PATHS
# ============================================================================

# --- Final Model Paths (after supervised training) ---
BI_ENCODER_PATH = MODELS_DIR / "bi-encoder"
CROSS_ENCODER_PATH = MODELS_DIR / "cross-encoder"
LIGHT_RERANKER_PATH = MODELS_DIR / "light-reranker"
PHOBERT_LAW_PATH = MODELS_DIR / "phobert-law"

# --- Unsupervised Adaptation Model Paths ---
DAPT_ADAPTED_MODEL_PATH = MODELS_DIR / "dapt_base_model"
TSDAE_ADAPTED_MODEL_PATH = MODELS_DIR / "tsdae_adapted_model"

# --- Index Paths ---
FAISS_INDEX_PATH = INDEXES_DIR / "legal.faiss"
INDEX_TO_AID_PATH = INDEXES_DIR / "index_to_aid.json"

# ============================================================================
# BASE MODEL NAMES (from HuggingFace)
# ============================================================================

BI_ENCODER_MODEL_NAME = get_env_var("LAWBOT_BI_ENCODER_MODEL_NAME", "bkai-foundation-models/vietnamese-bi-encoder")
CROSS_ENCODER_MODEL_NAME = get_env_var("LAWBOT_CROSS_ENCODER_MODEL_NAME", "vinai/phobert-base-v2")
LIGHT_RERANKER_MODEL_NAME = get_env_var("LAWBOT_LIGHT_RERANKER_MODEL_NAME", "vinai/phobert-base-v2")

# ============================================================================
# DOMAIN ADAPTATION & SYSTEM CONFIG
# ============================================================================

# --- Unsupervised Domain Adaptation ---
DAPT_MAX_LENGTH = get_env_int("LAWBOT_DAPT_MAX_LENGTH", 128)
DAPT_DATASET_SIZE_LIMIT = get_env_int("LAWBOT_DAPT_DATASET_SIZE_LIMIT", 10000)
TSDAE_EPOCHS = get_env_int("LAWBOT_TSDAE_EPOCHS", 1)
TSDAE_BATCH_SIZE = get_env_int("LAWBOT_TSDAE_BATCH_SIZE", 8)

# --- System & Error Handling ---
MAX_RETRIES = get_env_int("LAWBOT_MAX_RETRIES", 3)
RETRY_DELAY_SECONDS = get_env_int("LAWBOT_RETRY_DELAY_SECONDS", 30)
FORCE_CPU_MODE = get_env_bool("LAWBOT_FORCE_CPU_MODE", False)

# ============================================================================
# DATALOADER CONFIG
# ============================================================================
BI_ENCODER_DATALOADER_NUM_WORKERS = get_env_int("LAWBOT_BI_ENCODER_DATALOADER_NUM_WORKERS", 0)
BI_ENCODER_DATALOADER_PIN_MEMORY = get_env_bool("LAWBOT_BI_ENCODER_DATALOADER_PIN_MEMORY", True)
BI_ENCODER_DATALOADER_PREFETCH_FACTOR = get_env_int("LAWBOT_BI_ENCODER_DATALOADER_PREFETCH_FACTOR", 2)

CROSS_ENCODER_DATALOADER_NUM_WORKERS = get_env_int("LAWBOT_CROSS_ENCODER_DATALOADER_NUM_WORKERS", 0)
CROSS_ENCODER_DATALOADER_PIN_MEMORY = get_env_bool("LAWBOT_CROSS_ENCODER_DATALOADER_PIN_MEMORY", True)
CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR = get_env_int("LAWBOT_CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR", 2)

LIGHT_RERANKER_DATALOADER_NUM_WORKERS = get_env_int("LAWBOT_LIGHT_RERANKER_DATALOADER_NUM_WORKERS", 0)

# ============================================================================
# TRAINING HYPERPARAMETERS (BASE)
# ============================================================================

# --- Bi-Encoder ---
BI_ENCODER_LR = get_env_float("LAWBOT_BI_ENCODER_LR", 2e-5)
BI_ENCODER_WARMUP_RATIO = get_env_float("LAWBOT_BI_ENCODER_WARMUP_RATIO", 0.1)
BI_ENCODER_EVAL_STEPS = get_env_int("LAWBOT_BI_ENCODER_EVAL_STEPS", 250)
BI_ENCODER_EARLY_STOPPING_PATIENCE = get_env_int("LAWBOT_BI_ENCODER_EARLY_STOPPING_PATIENCE", 3)
BI_ENCODER_EARLY_STOPPING_THRESHOLD = get_env_float("LAWBOT_BI_ENCODER_EARLY_STOPPING_THRESHOLD", 0.005)
BI_ENCODER_GRADIENT_ACCUMULATION_STEPS = get_env_int("LAWBOT_BI_ENCODER_GRADIENT_ACCUMULATION_STEPS", 1)

# --- Cross-Encoder ---
CROSS_ENCODER_LR = get_env_float("LAWBOT_CROSS_ENCODER_LR", 2e-5)
CROSS_ENCODER_WARMUP_RATIO = get_env_float("LAWBOT_CROSS_ENCODER_WARMUP_RATIO", 0.1)
CROSS_ENCODER_EVAL_STEPS = get_env_int("LAWBOT_CROSS_ENCODER_EVAL_STEPS", 500)
CROSS_ENCODER_EARLY_STOPPING_PATIENCE = get_env_int("LAWBOT_CROSS_ENCODER_EARLY_STOPPING_PATIENCE", 3)
CROSS_ENCODER_EARLY_STOPPING_THRESHOLD = get_env_float("LAWBOT_CROSS_ENCODER_EARLY_STOPPING_THRESHOLD", 0.005)
CROSS_ENCODER_GRADIENT_ACCUMULATION_STEPS = get_env_int("LAWBOT_CROSS_ENCODER_GRADIENT_ACCUMULATION_STEPS", 2)

# --- Light Reranker ---
LIGHT_RERANKER_LR = get_env_float("LAWBOT_LIGHT_RERANKER_LR", 3e-5)
LIGHT_RERANKER_WARMUP_RATIO = get_env_float("LAWBOT_LIGHT_RERANKER_WARMUP_RATIO", 0.1)
LIGHT_RERANKER_EVAL_STEPS = get_env_int("LAWBOT_LIGHT_RERANKER_EVAL_STEPS", 250)
LIGHT_RERANKER_EARLY_STOPPING_PATIENCE = get_env_int("LAWBOT_LIGHT_RERANKER_EARLY_STOPPING_PATIENCE", 3)
LIGHT_RERANKER_EARLY_STOPPING_THRESHOLD = get_env_float("LAWBOT_LIGHT_RERANKER_EARLY_STOPPING_THRESHOLD", 0.005)
LIGHT_RERANKER_GRADIENT_ACCUMULATION_STEPS = get_env_int("LAWBOT_LIGHT_RERANKER_GRADIENT_ACCUMULATION_STEPS", 1)
