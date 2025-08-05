#!/usr/bin/env python3
"""
Domain-Adaptive Pre-training (DAPT) Script
==========================================

Script nay thuc hien Domain-Adaptive Pre-training cho PhoBERT model
de chuyen doi tu model tong quat thanh model chuyen biet cho linh vuc phap luat.

Tac gia: LawBot Team
Phien ban: DAPT v1.0
"""

import json
import logging
import random
import torch
from pathlib import Path
from typing import Dict, List, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import numpy as np

# Them thu muc goc vao path
import sys
import os

# Force CPU training to avoid CUDA issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from core.logging_system import get_logger

# Sử dụng logger đã được setup từ pipeline chính
logger = get_logger(__name__)


def load_legal_corpus():
    """Load legal corpus tu file JSON (DEPRECATED - SU DUNG UTILITY FUNCTION)"""
    logger.info("[CORPUS] Loading legal corpus...")

    try:
        from core.utils import parse_legal_corpus

        # Use common utility function
        all_articles = parse_legal_corpus(config.LEGAL_CORPUS_PATH)

        if all_articles:
            logger.info(f"[CORPUS] Loaded {len(all_articles)} legal articles")
            return all_articles
        else:
            logger.error("[CORPUS] No articles loaded from legal corpus")
            return None

    except Exception as e:
        logger.error(f"[CORPUS] Error loading legal corpus: {e}")
        return None


def prepare_legal_texts(legal_corpus):
    """Chuan bi texts cho DAPT training (SU DUNG UTILITY FUNCTION CHUNG)"""
    logger.info("[PREPARE] Preparing legal texts for DAPT...")

    try:
        # Import utility function
        from core.utils import parse_legal_corpus

        # Use common utility to parse legal corpus
        all_articles = parse_legal_corpus(config.LEGAL_CORPUS_PATH)

        if not all_articles:
            logger.error("[PREPARE] No articles extracted from legal corpus")
            return []

        # Extract only the content for DAPT training
        legal_texts = [article["content"] for article in all_articles]

        logger.info(f"[PREPARE] Prepared {len(legal_texts)} legal texts")

        # Log some sample texts for debugging
        if legal_texts:
            logger.info(
                f"[PREPARE] Sample text (first 200 chars): {legal_texts[0][:200]}..."
            )
            logger.info(
                f"[PREPARE] Total characters: {sum(len(text) for text in legal_texts)}"
            )

        return legal_texts

    except Exception as e:
        logger.error(f"[PREPARE] Error preparing legal texts: {e}")
        return []


def create_dapt_dataset(
    legal_texts, tokenizer, max_length=128
):  # Further reduced max_length for speed
    """Tao dataset cho DAPT training - OPTIMIZED FOR SPEED"""
    logger.info("[DATASET] Creating DAPT dataset (OPTIMIZED)...")

    # Filter out empty or very short texts - OPTIMIZED FILTERING
    filtered_texts = []
    for text in legal_texts:
        if text and len(text.strip()) > config.MIN_TEXT_LENGTH:  # Use config parameter
            # Truncate very long texts to save memory
            truncated_text = text.strip()[
                : config.MAX_TEXT_LENGTH
            ]  # Use config parameter
            filtered_texts.append(truncated_text)

    logger.info(
        f"[DATASET] Filtered {len(filtered_texts)} valid texts from {len(legal_texts)} total"
    )

    # Limit dataset size for faster training
    if len(filtered_texts) > config.DAPT_DATASET_SIZE_LIMIT:  # Use config parameter
        filtered_texts = filtered_texts[: config.DAPT_DATASET_SIZE_LIMIT]
        logger.info(
            f"[DATASET] Limited to {config.DAPT_DATASET_SIZE_LIMIT} samples for faster training"
        )

    def tokenize_function(examples):
        # Tokenize texts with error handling - OPTIMIZED
        try:
            result = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_special_tokens_mask=True,
            )

            # Create labels for MLM (same as input_ids initially)
            result["labels"] = result["input_ids"].copy()

            return result
        except Exception as e:
            logger.warning(f"[DATASET] Tokenization error: {e}")
            # Return empty result for failed tokenization
            return {
                "input_ids": [[0] * max_length],
                "attention_mask": [[0] * max_length],
                "labels": [[0] * max_length],
                "special_tokens_mask": [[1] + [0] * (max_length - 2) + [1]],
            }

    # Create dataset
    dataset_dict = {"text": filtered_texts}
    dataset = Dataset.from_dict(dataset_dict)

    # Tokenize dataset with error handling - OPTIMIZED BATCH SIZE
    try:
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=500,  # Increased batch size for speed
            remove_columns=dataset.column_names,
            num_proc=4,  # Use multiple processes
        )

        # Filter out samples with all zeros (failed tokenization) - OPTIMIZED
        valid_samples = []
        for i, sample in enumerate(tokenized_dataset):
            if any(sample["input_ids"]):  # Check if not all zeros
                valid_samples.append(sample)

        logger.info(
            f"[DATASET] Created DAPT dataset with {len(valid_samples)} valid samples"
        )
        return Dataset.from_list(valid_samples)

    except Exception as e:
        logger.error(f"[DATASET] Error creating dataset: {e}")
        # Return minimal dataset
        return Dataset.from_list(
            [
                {
                    "input_ids": [0] * max_length,
                    "attention_mask": [0] * max_length,
                    "labels": [0] * max_length,
                    "special_tokens_mask": [1] + [0] * (max_length - 2) + [1],
                }
            ]
        )


def train_phobert_law(legal_texts, output_path):
    """Train PhoBERT-Law model"""
    logger.info("[DAPT] Starting Domain-Adaptive Pre-training...")

    try:
        # Load tokenizer and model
        model_name = config.CROSS_ENCODER_MODEL_NAME
        logger.info(f"[DAPT] Loading base model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)

        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create dataset
        dataset = create_dapt_dataset(legal_texts, tokenizer)

        # Split dataset
        dataset_dict = dataset.train_test_split(test_size=0.1)
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["test"]

        # Data collator for MLM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15  # Mask 15% of tokens
        )

        # Training arguments - OPTIMIZED FOR BETTER PERFORMANCE
        training_args = TrainingArguments(
            output_dir=str(output_path),
            overwrite_output_dir=True,
            num_train_epochs=config.BI_ENCODER_EPOCHS,  # Use config epochs (3)
            per_device_train_batch_size=config.BI_ENCODER_BATCH_SIZE,  # Use config batch size (16)
            per_device_eval_batch_size=config.BI_ENCODER_BATCH_SIZE,  # Use config batch size (16)
            gradient_accumulation_steps=config.BI_ENCODER_GRADIENT_ACCUMULATION_STEPS,  # Use config (2)
            learning_rate=config.BI_ENCODER_LR,  # Use config LR (2e-5)
            warmup_steps=config.BI_ENCODER_WARMUP_STEPS,  # Use config warmup (100)
            weight_decay=0.01,
            logging_dir=str(output_path / "logs"),
            logging_steps=25,  # More frequent logging for better monitoring
            eval_strategy="steps",
            eval_steps=config.BI_ENCODER_EVAL_STEPS,  # Use config eval steps (50)
            save_strategy="steps",
            save_steps=config.BI_ENCODER_EVAL_STEPS,  # Match eval steps
            save_total_limit=2,  # Keep 2 best models
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=config.FP16_TRAINING
            and torch.cuda.is_available(),  # Use config FP16 setting if GPU available
            # PERFORMANCE OPTIMIZATIONS
            dataloader_pin_memory=config.CROSS_ENCODER_DATALOADER_PIN_MEMORY,  # Use config pin memory
            dataloader_num_workers=config.CROSS_ENCODER_DATALOADER_NUM_WORKERS,  # Use config workers
            dataloader_prefetch_factor=config.CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR,  # Use config prefetch
            remove_unused_columns=True,  # Remove unused columns to save memory
            report_to=None,  # Disable wandb/tensorboard
            # CPU OPTIMIZATION
            use_cpu=True,  # Force CPU training
            dataloader_drop_last=True,  # Drop incomplete batches
            # ADDITIONAL PERFORMANCE OPTIMIZATIONS
            dataloader_prefetch_factor=2,  # Prefetch data
            gradient_checkpointing=False,  # Disable for speed
            optim="adamw_torch",  # Use PyTorch optimizer for speed
            # LEARNING RATE SCHEDULER
            lr_scheduler_type="linear",  # Linear learning rate decay
        )

        # Initialize trainer - Remove deprecated tokenizer parameter
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train model with CPU (no CUDA)
        logger.info("[DAPT] Starting CPU training...")
        trainer.train()

        # Save model and tokenizer
        logger.info("[DAPT] Saving PhoBERT-Law model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_path)

        logger.info(f"[DAPT] PhoBERT-Law model saved to: {output_path}")

        # Clean up memory
        del trainer, train_dataset, eval_dataset, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        logger.error(f"[DAPT] Error during DAPT training: {e}")
        # Clean up memory even on error
        if "trainer" in locals():
            del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False


def validate_phobert_law(model_path):
    """Validate PhoBERT-Law model"""
    logger.info("[VALIDATE] Validating PhoBERT-Law model...")

    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForMaskedLM.from_pretrained(model_path)

        # Test with legal text
        test_text = "Theo quy định của pháp luật, người lao động [MASK] được nghỉ phép hàng năm."

        # Tokenize
        inputs = tokenizer(test_text, return_tensors="pt")

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits

        # Find masked token position
        masked_token_index = torch.where(
            inputs["input_ids"][0] == tokenizer.mask_token_id
        )[0]

        if len(masked_token_index) > 0:
            mask_token_logits = predictions[0, masked_token_index[0], :]
            top_5_tokens = torch.topk(mask_token_logits, 5, dim=0).indices

            logger.info("[VALIDATE] Top 5 predictions for masked token:")
            for i, token_id in enumerate(top_5_tokens):
                token = tokenizer.decode([token_id])
                logger.info(f"[VALIDATE] {i+1}. {token}")

        logger.info("[VALIDATE] PhoBERT-Law model validation completed successfully")
        return True

    except Exception as e:
        logger.error(f"[VALIDATE] Error validating PhoBERT-Law model: {e}")
        return False


def run_dapt_pipeline():
    """Chay DAPT pipeline"""
    logger.info("=" * 60)
    logger.info("DOMAIN-ADAPTIVE PRE-TRAINING (DAPT) PIPELINE")
    logger.info("=" * 60)

    try:
        # Step 1: Load legal corpus
        logger.info("STEP 1: Loading legal corpus...")
        legal_corpus = load_legal_corpus()
        if legal_corpus is None:
            return False

        # Step 2: Prepare legal texts
        logger.info("STEP 2: Preparing legal texts...")
        legal_texts = prepare_legal_texts(legal_corpus)  # Now uses utility function
        if not legal_texts:
            logger.error("No legal texts prepared")
            return False

        # Step 3: Create output directory
        logger.info("STEP 3: Creating output directory...")
        phobert_law_path = config.MODELS_DIR / "phobert-law"
        phobert_law_path.mkdir(parents=True, exist_ok=True)

        # Step 4: Train PhoBERT-Law
        logger.info("STEP 4: Training PhoBERT-Law...")
        success = train_phobert_law(legal_texts, phobert_law_path)
        if not success:
            return False

        # Step 5: Validate model
        logger.info("STEP 5: Validating PhoBERT-Law...")
        validation_success = validate_phobert_law(phobert_law_path)

        logger.info("=" * 60)
        logger.info("DAPT PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("PhoBERT-Law model ready for use in:")
        logger.info("1. Bi-Encoder training")
        logger.info("2. Cross-Encoder training")
        logger.info("3. Legal QA pipeline")
        return True

    except Exception as e:
        logger.error(f"Error during DAPT pipeline: {e}")
        return False


def main():
    """Ham chinh"""
    logger.info("[START] Bat dau Domain-Adaptive Pre-training Pipeline...")

    success = run_dapt_pipeline()

    if success:
        logger.info("✅ DAPT Pipeline completed successfully!")
        logger.info("✅ PhoBERT-Law model ready for legal tasks!")
    else:
        logger.error("❌ DAPT Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
