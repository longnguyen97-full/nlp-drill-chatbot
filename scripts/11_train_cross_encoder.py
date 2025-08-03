#!/usr/bin/env python3
"""
Train Cross-Encoder - Script Huan Luyen Cross-Encoder Toi Uu
============================================================

Script nay huan luyen Cross-Encoder voi du lieu da duoc fix mapping
va hyperparameters toi uu de tranh overfitting.

Tac gia: LawBot Team
Phien ban: Optimized Training v2.0
"""

import json
import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_pairs(file_path):
    """Load pairs tu file JSONL"""
    logger.info(f"Loading pairs from {file_path}...")

    pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pair = json.loads(line)
                pairs.append(pair)

    logger.info(f"Loaded {len(pairs)} pairs")
    return pairs


def compute_metrics(pred):
    """Tinh toan metrics cho evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def train_cross_encoder():
    """Huan luyen Cross-Encoder voi hyperparameters toi uu"""
    logger.info("=" * 60)
    logger.info("TRAIN CROSS-ENCODER - OPTIMIZED VERSION")
    logger.info("=" * 60)

    try:
        # Load pairs
        pairs_path = config.DATA_PROCESSED_DIR / "pairs.jsonl"
        pairs = load_pairs(pairs_path)

        if not pairs:
            logger.error("No pairs found. Please run data preparation first.")
            return False

        # Prepare dataset
        logger.info("Preparing dataset...")

        # Convert to format expected by transformers
        texts = []
        labels = []

        for pair in pairs:
            texts.append(pair["texts"])
            labels.append(pair["label"])

        # Create dataset
        dataset = Dataset.from_dict({"texts": texts, "labels": labels})

        # Split dataset
        dataset = dataset.train_test_split(test_size=0.15, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Eval samples: {len(eval_dataset)}")

        # Initialize model with proper configuration
        model_name = config.CROSS_ENCODER_MODEL_NAME
        logger.info(f"Initializing model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,  # Binary classification
            problem_type="single_label_classification",
        )

        # Add padding token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id

        # Preprocessing function
        def preprocess_function(examples):
            # Handle pairs format: {"texts": ["query", "passage"], "label": 0/1}
            # examples is a batch from HuggingFace Dataset
            texts = examples["texts"]

            # Extract queries and passages from the batch
            queries = [text[0] for text in texts]
            passages = [text[1] for text in texts]

            # Tokenize query-passage pairs
            result = tokenizer(
                queries,
                passages,
                truncation=True,
                padding="max_length",
                max_length=config.CROSS_ENCODER_MAX_LENGTH,
                return_tensors=None,  # Don't return tensors for dataset mapping
            )

            # Add labels if they exist
            if "label" in examples:
                result["labels"] = examples["label"]

            return result

        # Tokenize datasets
        train_dataset = train_dataset.map(
            preprocess_function, batched=True, remove_columns=["texts"]
        )
        eval_dataset = eval_dataset.map(
            preprocess_function, batched=True, remove_columns=["texts"]
        )

        # Training arguments (optimized)
        training_args = TrainingArguments(
            output_dir=str(config.CROSS_ENCODER_PATH),
            num_train_epochs=config.CROSS_ENCODER_EPOCHS,
            per_device_train_batch_size=config.CROSS_ENCODER_BATCH_SIZE,
            per_device_eval_batch_size=config.CROSS_ENCODER_BATCH_SIZE,
            warmup_steps=config.CROSS_ENCODER_WARMUP_STEPS,
            weight_decay=0.01,  # Add weight decay
            logging_dir="logs/cross-encoder",
            logging_steps=50,
            # Fix: Use eval_strategy instead of evaluation_strategy
            eval_strategy="steps",
            eval_steps=config.CROSS_ENCODER_EVAL_STEPS,
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            learning_rate=config.CROSS_ENCODER_LR,
            gradient_accumulation_steps=config.CROSS_ENCODER_GRADIENT_ACCUMULATION_STEPS,
            dataloader_num_workers=config.CROSS_ENCODER_DATALOADER_NUM_WORKERS,
            remove_unused_columns=False,
            report_to=None,
        )

        logger.info("Training parameters:")
        logger.info(f"  - Epochs: {training_args.num_train_epochs}")
        logger.info(f"  - Batch size: {training_args.per_device_train_batch_size}")
        logger.info(f"  - Learning rate: {training_args.learning_rate}")
        logger.info(f"  - Warmup steps: {training_args.warmup_steps}")
        logger.info(
            f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}"
        )

        # Create trainer with proper data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Train model
        logger.info("Starting training...")
        trainer.train()

        # Save model
        trainer.save_model()
        logger.info("Model saved successfully!")

        # Evaluate final model
        logger.info("Evaluating final model...")
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")

        # Save training info
        training_info = {
            "model_name": model_name,
            "pairs_count": len(pairs),
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate,
            "warmup_steps": training_args.warmup_steps,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "final_eval_results": eval_results,
            "optimization": "anti_overfitting",
        }

        training_info_path = config.CROSS_ENCODER_PATH / "training_info.json"
        with open(training_info_path, "w", encoding="utf-8") as f:
            json.dump(training_info, f, ensure_ascii=False, indent=2)

        logger.info("=" * 60)
        logger.info("CROSS-ENCODER TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Next steps:")
        logger.info("1. Run comprehensive evaluation")
        logger.info("2. Test pipeline performance")
        logger.info("3. Deploy if metrics are satisfactory")

        return True

    except Exception as e:
        logger.error(f"Error during Cross-Encoder training: {e}")
        return False


def main():
    """Main function"""
    success = train_cross_encoder()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
