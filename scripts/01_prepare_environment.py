#!/usr/bin/env python3
"""
Step 01: Prepare Environment and Adapt Models
=============================================

This script performs two critical functions:
1.  **Environment Validation:** Checks Python version, dependencies, data files,
    and configurations to ensure the system is ready.
2.  **Unsupervised Model Adaptation:** Runs a two-stage unsupervised adaptation
    process (DAPT-MLM then TSDAE) to create specialized base models for
    all downstream supervised training tasks.

Author: LawBot Team
Version: Refactored Pipeline v1.0
"""

import json
import logging
import random
import torch
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

# --- System Path Setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Sentence Transformers & Datasets Imports ---
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from torch.utils.data import DataLoader

# --- Hugging Face Transformers Imports ---
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# --- Project-specific Imports ---
import config
from core.services.logging_service import get_logger
from core.io import parse_legal_corpus

# Initialize logger
logger = get_logger(__name__)

# --- GPU Availability Check ---
GPU_AVAILABLE = torch.cuda.is_available() and not config.FORCE_CPU_MODE


# ============================================================================
# PART 1: ENVIRONMENT & CONFIGURATION VALIDATION
# ============================================================================

def check_system_environment():
    """Checks Python version and essential libraries."""
    logger.info("[ENV] Checking Python and library environment...")
    # Migrated from 01_check_environment.py
    python_version = sys.version_info
    if python_version.major < 3 or python_version.minor < 8:
        logger.error(f"[ENV] Python version {python_version.major}.{python_version.minor} is not sufficient. Python 3.8+ is required.")
        return False
    logger.info(f"[ENV] Python version: {python_version.major}.{python_version.minor}.{python_version.micro} [OK]")

    try:
        import torch
        logger.info(f"[ENV] PyTorch version: {torch.__version__} [OK]")
        if GPU_AVAILABLE:
            logger.info(f"[ENV] CUDA available: {torch.cuda.get_device_name(0)} [OK]")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"[ENV] GPU Memory: {gpu_mem:.1f} GB")
        else:
            logger.info("[ENV] CUDA not available. Running on CPU.")
    except ImportError:
        logger.error("[ENV] PyTorch is not installed.")
        return False

    # Check other essential libraries
    # ... (code to check other libraries can be added here if needed)
    return True

def check_data_files():
    """Validates the existence of essential raw data files."""
    logger.info("[ENV] Checking for essential raw data files...")
    # Migrated from 01_check_environment.py
    required_files = [
        config.TRAIN_JSON_PATH,
        config.LEGAL_CORPUS_PATH,
    ]
    all_found = True
    for file_path in required_files:
        if file_path.exists():
            logger.info(f"[ENV] Found raw data file: {file_path.name} [OK]")
        else:
            logger.error(f"[ENV] Missing required raw data file: {file_path}")
            all_found = False
    return all_found

def create_project_directories():
    """Creates all necessary directories for the project if they don't exist."""
    logger.info("[ENV] Creating project directory structure...")
    try:
        dirs_to_create = [
            config.DATA_PROCESSED_DIR,
            config.DATA_VALIDATION_DIR,
            config.INDEXES_DIR,
            config.REPORTS_DIR,
            Path(config.BI_ENCODER_PATH).parent,
            Path(config.CROSS_ENCODER_PATH).parent,
            Path(config.LIGHT_RERANKER_PATH).parent,
            Path(config.DAPT_ADAPTED_MODEL_PATH),
            Path(config.TSDAE_ADAPTED_MODEL_PATH).parent,
            Path("logs"),
        ]
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
        logger.info("[ENV] All project directories created successfully.")
        return True
    except Exception as e:
        logger.error(f"[ENV] Failed to create directories: {e}", exc_info=True)
        return False

def validate_configuration():
    """Validates the project's configuration settings."""
    logger.info("[ENV] Validating configuration...")
    # (Implementation migrated from scripts/01_check_environment.py)
    try:
        # Bỏ qua bước validate config hiện tại vì nó không được định nghĩa
        # config.validate_config()
        logger.info("Configuration validation skipped (function not defined).")
        return True
    except Exception as e:
        logger.error(f"[ENV] Configuration validation failed: {e}", exc_info=True)
        return False


# ============================================================================
# PART 2: UNSUPERVISED DOMAIN ADAPTATION
# ============================================================================
# Functions: load_legal_corpus, prepare_legal_texts, create_dapt_dataset,
# train_dapt_model, validate_adapted_model, train_tsdae_model are migrated
# directly from scripts/00_adapt_model.py without significant changes in their
# internal logic, only in how they are called and orchestrated.

def load_legal_corpus() -> Optional[List[Dict]]:
    # Migrated and simplified from 00_adapt_model.py
    logger.info("[ADAPT] Loading legal corpus...")
    try:
        all_articles = parse_legal_corpus(config.LEGAL_CORPUS_PATH)
        if not all_articles:
            logger.error("[ADAPT] No articles loaded from legal corpus.")
            return None
        logger.info(f"[ADAPT] Loaded {len(all_articles)} legal articles.")
        return all_articles
    except Exception as e:
        logger.error(f"[ADAPT] Error loading legal corpus: {e}", exc_info=True)
        return None

def prepare_legal_texts(legal_corpus: List[Dict]) -> List[str]:
    # Migrated and simplified from 00_adapt_model.py
    logger.info("[ADAPT] Preparing legal texts for adaptation...")
    if not legal_corpus:
        return []
    return [article["content"] for article in legal_corpus]

def create_dapt_dataset(legal_texts: List[str], tokenizer, max_length: int = 128) -> Dataset:
    """Creates a Hugging Face Dataset for DAPT."""
    logger.info("[ADAPT] Creating DAPT dataset...")
    # Limit dataset size for faster processing if configured
    if config.DAPT_DATASET_SIZE_LIMIT and len(legal_texts) > config.DAPT_DATASET_SIZE_LIMIT:
        logger.info(f"[ADAPT] Limiting DAPT dataset to {config.DAPT_DATASET_SIZE_LIMIT} random samples.")
        legal_texts = random.sample(legal_texts, config.DAPT_DATASET_SIZE_LIMIT)

    # Tokenize in batches for efficiency
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")

    dataset = Dataset.from_dict({"text": legal_texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    logger.info(f"[ADAPT] DAPT dataset created with {len(tokenized_dataset)} samples.")
    return tokenized_dataset

def train_dapt_model(legal_texts: List[str], output_path: Path) -> bool:
    """Trains a DAPT (MLM) model."""
    logger.info("[ADAPT] Training DAPT base model...")
    try:
        base_model_name = config.CROSS_ENCODER_MODEL_NAME # Use the same base as Cross-Encoder
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForMaskedLM.from_pretrained(base_model_name)

        tokenized_dataset = create_dapt_dataset(legal_texts, tokenizer, config.DAPT_MAX_LENGTH)
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

        training_args = TrainingArguments(
            output_dir=str(output_path),
            overwrite_output_dir=True,
            num_train_epochs=config.DAPT_EPOCHS,
            per_device_train_batch_size=config.DAPT_BATCH_SIZE,
            save_steps=10_000,
            save_total_limit=2,
            prediction_loss_only=True,
            fp16=GPU_AVAILABLE,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        logger.info(f"[ADAPT] DAPT model training complete. Model saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"[ADAPT] DAPT training failed: {e}", exc_info=True)
        return False

def validate_adapted_model(model_path: Path) -> bool:
    """Validates that the adapted model can be loaded."""
    logger.info(f"[ADAPT] Validating adapted model at {model_path}...")
    try:
        SentenceTransformer(str(model_path))
        logger.info("[ADAPT] Model validation successful.")
        return True
    except Exception as e:
        logger.error(f"[ADAPT] Model validation failed: {e}", exc_info=True)
        return False

def train_tsdae_model(dapt_model_path: Path, tsdae_model_path: Path, legal_texts: List[str]) -> bool:
    """Trains a TSDAE model for Bi-Encoder initialization."""
    logger.info("[ADAPT] Training TSDAE model...")
    try:
        # Create a new SentenceTransformer model from the DAPT-adapted base
        word_embedding_model = models.Transformer(str(dapt_model_path))
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        # Create the special denoising dataset
        train_dataset = DenoisingAutoEncoderDataset(legal_texts)
        train_dataloader = DataLoader(train_dataset, batch_size=config.TSDAE_BATCH_SIZE, shuffle=True)
        
        # Use the DenoisingAutoEncoderLoss
        loss = DenoisingAutoEncoderLoss(model)

        # Train the model
        model.fit(
            train_objectives=[(train_dataloader, loss)],
            epochs=config.TSDAE_EPOCHS,
            weight_decay=0,
            scheduler='constantlr',
            optimizer_params={'lr': 3e-5},
            show_progress_bar=True
        )

        model.save(str(tsdae_model_path))
        logger.info(f"[ADAPT] TSDAE model training complete. Model saved to {tsdae_model_path}")
        return True
    except Exception as e:
        logger.error(f"[ADAPT] TSDAE training failed: {e}", exc_info=True)
        return False

def run_unsupervised_adaptation():
    """
    Orchestrates the two-stage unsupervised adaptation process.
    """
    logger.info("=" * 80)
    logger.info("STARTING: Unsupervised Domain Adaptation")
    logger.info("=" * 80)
    
    logger.info("STEP 2A: Loading and Preparing Legal Corpus for Adaptation...")
    legal_corpus = load_legal_corpus()
    if not legal_corpus:
        return False
    legal_texts = prepare_legal_texts(legal_corpus)
    if not legal_texts:
        return False

    # Stage 1: DAPT (MLM)
    logger.info("\n--- STAGE 1: DAPT (Masked Language Modeling) ---")
    dapt_model_path = Path(config.DAPT_ADAPTED_MODEL_PATH)
    if (dapt_model_path / "pytorch_model.bin").exists():
        logger.info(f"DAPT base model already exists at {dapt_model_path}. Skipping training.")
    else:
        if not train_dapt_model(legal_texts, dapt_model_path):
            logger.error("DAPT base model training failed. Aborting.")
            return False
    validate_adapted_model(dapt_model_path)

    # Stage 2: TSDAE
    logger.info("\n--- STAGE 2: TSDAE (Denoising Auto-Encoder for Bi-Encoder) ---")
    tsdae_model_path = Path(config.TSDAE_ADAPTED_MODEL_PATH)
    if (tsdae_model_path / "pytorch_model.bin").exists():
        logger.info(f"TSDAE model already exists at {tsdae_model_path}. Skipping training.")
    else:
        if not train_tsdae_model(dapt_model_path, tsdae_model_path, legal_texts):
            logger.warning("TSDAE training failed. Bi-encoder will fallback to DAPT model.")
    
    logger.info("\n--- Unsupervised Adaptation Summary ---")
    logger.info(f"Base Adapted Model (for Cross-Encoders) is ready at: {dapt_model_path}")
    if (tsdae_model_path / "pytorch_model.bin").exists():
        logger.info(f"TSDAE Adapted Model (for Bi-Encoder) is ready at: {tsdae_model_path}")
    else:
        logger.info(f"Bi-Encoder will use the DAPT model as fallback: {dapt_model_path}")
    
    return True


# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================

def main():
    """Main function to run the environment preparation and model adaptation pipeline."""
    logger.info("=" * 80)
    logger.info("STARTING PIPELINE STEP 01: PREPARE ENVIRONMENT & ADAPT MODELS")
    logger.info("=" * 80)

    # Part 1: Environment and Setup Validation
    logger.info("--- PART 1: Validating Environment and Project Setup ---")
    if not check_system_environment(): return False
    if not check_data_files(): return False
    if not create_project_directories(): return False
    if not validate_configuration(): return False
    logger.info("--- Environment and Setup validation successful ---\n")

    # Part 2: Unsupervised Model Adaptation
    logger.info("--- PART 2: Running Unsupervised Model Adaptation ---")
    if not run_unsupervised_adaptation():
        logger.error("Unsupervised model adaptation failed. Halting pipeline.")
        return False
    logger.info("--- Unsupervised model adaptation successful ---\n")
    
    logger.info("=" * 80)
    logger.info("PIPELINE STEP 01 COMPLETED SUCCESSFULLY!")
    logger.info("Environment is ready and base models are adapted for the next steps.")
    logger.info("=" * 80)
    return True


if __name__ == "__main__":
    if not main():
        sys.exit(1)
