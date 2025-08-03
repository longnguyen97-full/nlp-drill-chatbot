#!/usr/bin/env python3
"""
Train Bi-Encoder - Script Huan Luyen Bi-Encoder Toi Uu
=======================================================

Script nay huan luyen Bi-Encoder voi du lieu da duoc fix mapping
va hyperparameters toi uu de tranh overfitting.

Tac gia: LawBot Team
Phien ban: Optimized Training v2.0
"""

import json
import logging
import torch
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_triplets(file_path):
    """Load triplets tu file JSONL"""
    logger.info(f"Loading triplets from {file_path}...")

    triplets = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                triplet = json.loads(line)
                triplets.append(triplet)

    logger.info(f"Loaded {len(triplets)} triplets")
    return triplets


def create_training_examples(triplets):
    """Tao training examples tu triplets"""
    logger.info("Creating training examples...")

    train_examples = []
    for triplet in triplets:
        # Positive example
        train_examples.append(
            InputExample(texts=[triplet["query"], triplet["positive"]], label=1.0)
        )
        # Negative example
        train_examples.append(
            InputExample(texts=[triplet["query"], triplet["negative"]], label=0.0)
        )

    logger.info(f"Created {len(train_examples)} training examples")
    return train_examples


def train_bi_encoder():
    """Huan luyen Bi-Encoder voi hyperparameters toi uu"""
    logger.info("=" * 60)
    logger.info("TRAIN BI-ENCODER - OPTIMIZED VERSION")
    logger.info("=" * 60)

    try:
        # Load triplets
        triplets_path = config.DATA_PROCESSED_DIR / "triplets.jsonl"
        triplets = load_triplets(triplets_path)

        if not triplets:
            logger.error("No triplets found. Please run data preparation first.")
            return False

        # Create training examples
        train_examples = create_training_examples(triplets)

        # Initialize model
        model_name = config.BI_ENCODER_MODEL_NAME
        logger.info(f"Initializing model: {model_name}")

        model = SentenceTransformer(model_name)

        # Setup training parameters (optimized)
        train_dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=config.BI_ENCODER_BATCH_SIZE
        )

        # Use MultipleNegativesRankingLoss for better training
        train_loss = losses.MultipleNegativesRankingLoss(model)

        # Training parameters (optimized to prevent overfitting)
        num_epochs = config.BI_ENCODER_EPOCHS
        warmup_steps = config.BI_ENCODER_WARMUP_STEPS
        learning_rate = config.BI_ENCODER_LR

        logger.info(f"Training parameters:")
        logger.info(f"  - Epochs: {num_epochs}")
        logger.info(f"  - Batch size: {config.BI_ENCODER_BATCH_SIZE}")
        logger.info(f"  - Learning rate: {learning_rate}")
        logger.info(f"  - Warmup steps: {warmup_steps}")

        # Create output directory
        output_dir = config.BI_ENCODER_PATH
        output_dir.mkdir(parents=True, exist_ok=True)

        # Train model with optimized settings
        logger.info("Starting training...")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": learning_rate},
            show_progress_bar=True,
            checkpoint_path=str(output_dir / "checkpoint"),
            checkpoint_save_steps=100,
            checkpoint_save_total_limit=2,
        )

        # Save model
        model.save(str(output_dir))
        logger.info(f"Model saved to: {output_dir}")

        # Save training info
        training_info = {
            "model_name": model_name,
            "triplets_count": len(triplets),
            "training_examples": len(train_examples),
            "epochs": num_epochs,
            "batch_size": config.BI_ENCODER_BATCH_SIZE,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "loss_type": "MultipleNegativesRankingLoss",
            "optimization": "anti_overfitting",
        }

        with open(output_dir / "training_info.json", "w", encoding="utf-8") as f:
            json.dump(training_info, f, ensure_ascii=False, indent=2)

        logger.info("=" * 60)
        logger.info("BI-ENCODER TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Next steps:")
        logger.info("1. Build FAISS index")
        logger.info("2. Train Cross-Encoder")
        logger.info("3. Run evaluation")

        return True

    except Exception as e:
        logger.error(f"Error during Bi-Encoder training: {e}")
        return False


def main():
    """Main function"""
    success = train_bi_encoder()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
