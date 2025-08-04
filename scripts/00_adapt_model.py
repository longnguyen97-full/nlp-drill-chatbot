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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from core.logging_utils import get_logger

# Sử dụng logger đã được setup từ pipeline chính
logger = get_logger(__name__)


def load_legal_corpus():
    """Load legal corpus tu file JSON"""
    logger.info("[CORPUS] Loading legal corpus...")

    try:
        with open(config.LEGAL_CORPUS_PATH, "r", encoding="utf-8") as f:
            legal_corpus = json.load(f)

        logger.info(f"[CORPUS] Loaded {len(legal_corpus)} legal documents")
        return legal_corpus

    except Exception as e:
        logger.error(f"[CORPUS] Error loading legal corpus: {e}")
        return None


def prepare_legal_texts(legal_corpus):
    """Chuan bi texts cho DAPT training"""
    logger.info("[PREPARE] Preparing legal texts for DAPT...")

    legal_texts = []
    processed_count = 0
    error_count = 0

    for doc in legal_corpus:
        try:
            # Extract text content from legal documents
            if isinstance(doc, dict):
                # Handle different document formats
                if "content" in doc:
                    content = doc["content"]
                    # Check if content is a list (multiple articles)
                    if isinstance(content, list):
                        # Extract text from each article in the list
                        text_parts = []
                        for article in content:
                            if isinstance(article, dict):
                                if "content_Article" in article:
                                    text_parts.append(article["content_Article"])
                                elif "content" in article:
                                    text_parts.append(article["content"])
                                elif "text" in article:
                                    text_parts.append(article["text"])
                            else:
                                text_parts.append(str(article))
                        text = " ".join(text_parts)
                    else:
                        # Content is already a string
                        text = str(content)
                elif "text" in doc:
                    text = doc["text"]
                elif "article" in doc:
                    text = doc["article"]
                else:
                    # Try to concatenate all string values
                    text = " ".join(
                        [str(v) for v in doc.values() if isinstance(v, str)]
                    )
            else:
                text = str(doc)

            # Clean and normalize text
            text = text.strip()
            if len(text) > 50:  # Only include substantial texts
                legal_texts.append(text)
                processed_count += 1

        except Exception as e:
            error_count += 1
            logger.warning(f"[PREPARE] Error processing document {error_count}: {e}")
            continue

    logger.info(
        f"[PREPARE] Processed {processed_count} documents, {error_count} errors"
    )
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


def create_dapt_dataset(legal_texts, tokenizer, max_length=512):
    """Tao dataset cho DAPT training"""
    logger.info("[DATASET] Creating DAPT dataset...")

    def tokenize_function(examples):
        # Tokenize texts
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

    # Create dataset
    dataset_dict = {"text": legal_texts}
    dataset = Dataset.from_dict(dataset_dict)

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )

    logger.info(f"[DATASET] Created DAPT dataset with {len(tokenized_dataset)} samples")
    return tokenized_dataset


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

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_path),
            overwrite_output_dir=True,
            num_train_epochs=3,  # DAPT typically uses 3-5 epochs
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=str(output_path / "logs"),
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # Train model
        logger.info("[DAPT] Starting training...")
        trainer.train()

        # Save model and tokenizer
        logger.info("[DAPT] Saving PhoBERT-Law model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_path)

        logger.info(f"[DAPT] PhoBERT-Law model saved to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"[DAPT] Error during DAPT training: {e}")
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
        legal_texts = prepare_legal_texts(legal_corpus)
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
