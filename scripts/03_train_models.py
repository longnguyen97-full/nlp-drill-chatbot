#!/usr/bin/env python3
"""
Model Training & Evaluation Pipeline - Script Toi Uu Can Bang
============================================================

Script nay huan luyen Bi-Encoder, build FAISS index, huan luyen Cross-Encoder,
va evaluation trong mot buoc toi uu can bang giua hieu qua va de hieu.
PHAN MEM DA TICH HOP CHECKPOINT DE CO THE KHOI DONG LAI.

Tac gia: LawBot Team
Phien ban: Balanced Optimized Pipeline v7.0 (with Checkpointing)
"""

import json
import logging
import torch
import faiss
import numpy as np
import random
import pickle
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import sys

sys.path.append(str(Path(__file__).parent.parent))
import config
from core.logging_system import get_logger
from core.utils import (
    create_reranker_preprocess_function,
    create_light_reranker_preprocess_function,
)
from core.pipeline import LegalQAPipeline
from core.evaluation_reporter import BatchEvaluator, EvaluationReporter

# Sử dụng logger đã được setup từ pipeline chính
logger = get_logger(__name__)

# --- PHAN CHECKPOINT ---
CHECKPOINT_FILE = config.DATA_PROCESSED_DIR / "pipeline_checkpoint.json"


def load_checkpoint():
    """Tải trạng thái pipeline từ file checkpoint."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                logger.info(f"[CHECKPOINT] Found checkpoint file at {CHECKPOINT_FILE}")
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(
                f"[CHECKPOINT] Could not read checkpoint file: {e}. Starting fresh."
            )
            return {}
    return {}


def save_checkpoint(state):
    """Lưu trạng thái pipeline vào file checkpoint."""
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"[CHECKPOINT] Saved checkpoint: {state}")
    except IOError as e:
        logger.error(f"[CHECKPOINT] Could not save checkpoint file: {e}")


def is_step_complete(state, step_name):
    """Kiểm tra xem một bước đã hoàn thành chưa."""
    return state.get(step_name, False)


def mark_step_complete(state, step_name):
    """Đánh dấu một bước đã hoàn thành."""
    state[step_name] = True
    save_checkpoint(state)


# -------------------------


def load_prepared_training_data():
    """Load du lieu training da duoc chuan bi tu buoc truoc với error handling"""
    logger.info("[LOAD] Loading prepared training data...")

    # Load Bi-Encoder data
    bi_encoder_path = config.DATA_PROCESSED_DIR / "bi_encoder_train_optimized.jsonl"
    if not bi_encoder_path.exists():
        logger.error(
            "Bi-Encoder training data not found. Please run training data preparation pipeline first."
        )
        return None, None

    bi_encoder_data = []
    bi_encoder_errors = 0
    try:
        with open(bi_encoder_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    bi_encoder_data.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"[LOAD] Bi-Encoder line {i}: JSON decode error - {e}"
                    )
                    bi_encoder_errors += 1
                except Exception as e:
                    logger.warning(
                        f"[LOAD] Bi-Encoder line {i}: Unexpected error - {e}"
                    )
                    bi_encoder_errors += 1
    except Exception as e:
        logger.error(f"[LOAD] Error reading Bi-Encoder file: {e}")
        return None, None

    # Load Cross-Encoder data
    cross_encoder_path = (
        config.DATA_PROCESSED_DIR / "cross_encoder_train_optimized.jsonl"
    )
    if not cross_encoder_path.exists():
        logger.error(
            "Cross-Encoder training data not found. Please run training data preparation pipeline first."
        )
        return None, None

    cross_encoder_data = []
    cross_encoder_errors = 0
    try:
        with open(cross_encoder_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    cross_encoder_data.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"[LOAD] Cross-Encoder line {i}: JSON decode error - {e}"
                    )
                    cross_encoder_errors += 1
                except Exception as e:
                    logger.warning(
                        f"[LOAD] Cross-Encoder line {i}: Unexpected error - {e}"
                    )
                    cross_encoder_errors += 1
    except Exception as e:
        logger.error(f"[LOAD] Error reading Cross-Encoder file: {e}")
        return None, None

    logger.info(
        f"[LOAD] Loaded {len(bi_encoder_data)} Bi-Encoder samples (errors: {bi_encoder_errors})"
    )
    logger.info(
        f"[LOAD] Loaded {len(cross_encoder_data)} Cross-Encoder samples (errors: {cross_encoder_errors})"
    )

    # Validate data quality
    if len(bi_encoder_data) == 0:
        logger.error("[LOAD] No valid Bi-Encoder data loaded")
        return None, None

    if len(cross_encoder_data) == 0:
        logger.error("[LOAD] No valid Cross-Encoder data loaded")
        return None, None

    return bi_encoder_data, cross_encoder_data


def create_training_examples(triplets):
    """Tao training examples cho Bi-Encoder với error handling"""
    logger.info("[EXAMPLE] Creating training examples for Bi-Encoder...")

    examples = []
    skipped_count = 0

    for i, triplet in enumerate(triplets):
        try:
            # Validate triplet structure
            if not isinstance(triplet, dict):
                logger.warning(f"[SKIP] Triplet {i+1}: Not a dictionary")
                skipped_count += 1
                continue

            required_keys = ["anchor", "positive", "negative"]
            if not all(key in triplet for key in required_keys):
                missing_keys = [key for key in required_keys if key not in triplet]
                logger.warning(f"[SKIP] Triplet {i+1}: Missing keys {missing_keys}")
                skipped_count += 1
                continue

            # Ensure texts are strings and not empty
            anchor = str(triplet["anchor"]) if triplet["anchor"] is not None else ""
            positive = (
                str(triplet["positive"]) if triplet["positive"] is not None else ""
            )
            negative = (
                str(triplet["negative"]) if triplet["negative"] is not None else ""
            )

            # Skip if any text is empty
            if not anchor.strip() or not positive.strip() or not negative.strip():
                logger.warning(f"[SKIP] Triplet {i+1}: Empty text content")
                skipped_count += 1
                continue

            # Create positive example
            examples.append(InputExample(texts=[anchor, positive], label=1.0))

            # Create negative example
            examples.append(InputExample(texts=[anchor, negative], label=0.0))

        except Exception as e:
            logger.warning(f"[SKIP] Triplet {i+1}: Error processing - {e}")
            skipped_count += 1
            continue

    logger.info(
        f"[EXAMPLE] Created {len(examples)} training examples, skipped {skipped_count}"
    )

    if len(examples) == 0:
        logger.error("[ERROR] No valid examples for Bi-Encoder training")
        return []

    return examples


def train_bi_encoder_optimized(bi_encoder_data):
    """Huan luyen Bi-Encoder toi uu (DA SUA LOI VA THEM EVALUATOR)"""
    logger.info("[TRAIN] Training Bi-Encoder...")

    # Validate input data
    if bi_encoder_data is None:
        logger.error("[TRAIN] Bi-Encoder data is None")
        return None

    if len(bi_encoder_data) < 5:
        logger.error(
            f"[TRAIN] Bi-Encoder data too small: {len(bi_encoder_data)} samples"
        )
        return None

    logger.info(
        f"[TRAIN] Starting Bi-Encoder training with {len(bi_encoder_data)} samples"
    )

    # --- BAT DAU PHAN SUA LOI ---

    # Step 1: Tach du lieu training va validation
    # Ta se dung 10% du lieu lam validation de theo doi hieu suat mo hinh
    if len(bi_encoder_data) < 10:
        logger.error(
            f"Not enough data for training. Only {len(bi_encoder_data)} samples available."
        )
        return None

    random.shuffle(bi_encoder_data)
    train_size = int(len(bi_encoder_data) * 0.9)
    train_triplets = bi_encoder_data[:train_size]
    val_triplets = bi_encoder_data[train_size:]

    logger.info(
        f"Tach du lieu: {len(train_triplets)} samples for training, {len(val_triplets)} for validation."
    )

    # Validate split sizes
    if len(train_triplets) < 5:
        logger.error(f"Training set too small: {len(train_triplets)} samples")
        return None
    if len(val_triplets) < 2:
        logger.warning(f"Validation set very small: {len(val_triplets)} samples")

    # Create training examples
    train_examples = create_training_examples(train_triplets)
    if not train_examples:
        logger.error("No training examples created for Bi-Encoder. Cannot proceed.")
        return None

    # Create validation examples cho evaluator
    val_examples = []
    val_errors = 0
    for triplet in val_triplets:
        try:
            anchor = str(triplet.get("anchor", ""))
            positive = str(triplet.get("positive", ""))
            negative = str(triplet.get("negative", ""))
            if anchor.strip() and positive.strip() and negative.strip():
                val_examples.append(InputExample(texts=[anchor, positive], label=1.0))
                val_examples.append(InputExample(texts=[anchor, negative], label=0.0))
        except Exception as e:
            val_errors += 1
            continue

    logger.info(
        f"Created {len(val_examples)} validation examples (errors: {val_errors})"
    )

    # Initialize model using config
    try:
        logger.info(f"[MODEL] Loading base model: {config.BI_ENCODER_MODEL_NAME}")
        model = SentenceTransformer(config.BI_ENCODER_MODEL_NAME)
        logger.info(
            f"[MODEL] Model loaded successfully. Embedding dimension: {model.get_sentence_embedding_dimension()}"
        )
    except Exception as e:
        logger.error(f"[MODEL] Failed to load model: {e}")
        return None

    # Step 2: Tao Evaluator
    # Evaluator se tinh toan do tuong dong cosine giua cac cap cau trong tap validation
    # va so sanh voi nhan (1.0 cho cap positive, 0.0 cho cap negative)
    # Giup theo doi xem model co dang hoc dung huong hay khong
    if val_examples:
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            val_examples,
            name="bi-encoder-val",
            main_similarity=None,  # Use default cosine similarity
        )
        logger.info("Created EmbeddingSimilarityEvaluator for validation.")
    else:
        evaluator = None
        logger.warning("No validation data available, training without evaluator.")

    # --- KET THUC PHAN SUA LOI ---

    # Create data loader with Windows-compatible multiprocessing settings
    num_workers = 0 if os.name == "nt" else config.BI_ENCODER_DATALOADER_NUM_WORKERS
    prefetch_factor = (
        None if num_workers == 0 else config.BI_ENCODER_DATALOADER_PREFETCH_FACTOR
    )

    logger.info(
        f"[DATALOADER] Creating DataLoader with batch_size={config.BI_ENCODER_BATCH_SIZE}, num_workers={num_workers}"
    )

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config.BI_ENCODER_BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=config.BI_ENCODER_DATALOADER_PIN_MEMORY,
        prefetch_factor=prefetch_factor,
    )

    logger.info(
        f"[DATALOADER] DataLoader created successfully. Total batches: {len(train_dataloader)}"
    )

    # Setup loss function
    try:
        train_loss = losses.ContrastiveLoss(model)
        logger.info("[LOSS] ContrastiveLoss initialized successfully")
    except Exception as e:
        logger.error(f"[LOSS] Failed to initialize loss function: {e}")
        return None

    # Train model with optimized and CORRECTED parameters
    try:
        logger.info(
            f"[TRAIN] Starting Bi-Encoder training with {len(train_examples)} examples..."
        )
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=config.BI_ENCODER_EPOCHS,
            warmup_steps=config.BI_ENCODER_WARMUP_STEPS,
            show_progress_bar=True,
            optimizer_params={"lr": config.BI_ENCODER_LR},
            scheduler="WarmupLinear",
            weight_decay=0.01,
            # --- SUA LOI TAI DAY ---
            # 1. Doi ten tham so 'eval_steps' -> 'evaluation_steps'
            # 2. Them 'evaluator' de mo hinh co du lieu de danh gia
            evaluation_steps=config.BI_ENCODER_EVAL_STEPS,
            evaluator=evaluator,
            output_path=str(config.MODELS_DIR / "bi_encoder_optimized"),
            save_best_model=(
                True if evaluator else False
            ),  # Luu model tot nhat neu co evaluator
            # Additional optimizations
            checkpoint_path=str(
                config.MODELS_DIR / "bi_encoder_optimized" / "checkpoints"
            ),
            checkpoint_save_steps=config.BI_ENCODER_EVAL_STEPS,
            checkpoint_save_total_limit=3,  # Keep only 3 best checkpoints
        )
        logger.info("[TRAIN] Bi-Encoder training completed successfully!")

        # Log final training stats
        if evaluator:
            logger.info("[EVAL] Running final evaluation...")
            try:
                final_score = evaluator(model)
                logger.info(f"[EVAL] Final evaluation score: {final_score}")
            except Exception as eval_error:
                logger.warning(f"[EVAL] Final evaluation failed: {eval_error}")

    except Exception as e:
        logger.error(f"[TRAIN] Bi-Encoder training failed: {e}")
        # Try to save partial model
        try:
            model_path = config.MODELS_DIR / "bi_encoder_optimized"
            model.save(str(model_path))
            logger.info(f"[SAVE] Partial model saved to: {model_path}")
        except Exception as save_error:
            logger.error(f"[SAVE] Failed to save partial model: {save_error}")
        raise

    # Save model (luu lai lan cuoi, mac du fit() da luu best model)
    try:
        model_path = config.MODELS_DIR / "bi_encoder_optimized"
        model.save(str(model_path))
        logger.info(f"[SAVE] Bi-Encoder saved to: {model_path}")

        # Log training summary
        logger.info("=" * 50)
        logger.info("BI-ENCODER TRAINING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Training examples: {len(train_examples)}")
        logger.info(f"Validation examples: {len(val_examples)}")
        logger.info(f"Epochs: {config.BI_ENCODER_EPOCHS}")
        logger.info(f"Batch size: {config.BI_ENCODER_BATCH_SIZE}")
        logger.info(f"Learning rate: {config.BI_ENCODER_LR}")
        logger.info(f"Warmup steps: {config.BI_ENCODER_WARMUP_STEPS}")
        logger.info(f"Evaluation steps: {config.BI_ENCODER_EVAL_STEPS}")
        logger.info(f"Model saved to: {model_path}")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"[SAVE] Failed to save final model: {e}")
        raise

    return model


def build_faiss_index_optimized(model):
    """Build FAISS index toi uu (DA SUA LOI LOGIC PARSE DU LIEU)"""
    logger.info("[INDEX] Building FAISS index...")

    try:
        # Import utility function
        from core.utils import parse_legal_corpus

        # Parse legal corpus using common utility
        all_articles = parse_legal_corpus(config.LEGAL_CORPUS_PATH)

        if not all_articles:
            logger.error(
                "[INDEX] No articles extracted from legal corpus. Please check the corpus structure."
            )
            return False

        # Extract documents and aids
        documents = [article["content"] for article in all_articles]
        aids = [article["aid"] for article in all_articles]

        logger.info(f"[INDEX] Creating embeddings for {len(documents)} documents...")

        # Create embeddings with optimized batch size and performance settings
        embeddings = model.encode(
            documents,
            show_progress_bar=True,
            batch_size=config.BI_ENCODER_BATCH_SIZE,
            convert_to_numpy=True,
            device=(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),  # Use GPU if available
            normalize_embeddings=True,  # Normalize embeddings for better performance
        )

        if embeddings.size == 0:
            logger.error("[INDEX] No embeddings generated. Cannot build index.")
            return False

        # Build FAISS index
        dimension = embeddings.shape[1]
        faiss.normalize_L2(embeddings)

        # Use IndexFlatIP for simplicity and reliability
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype("float32"))

        # Save index
        config.INDEXES_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(config.FAISS_INDEX_PATH))

        # Save index-to-aid mapping
        with open(config.INDEX_TO_AID_PATH, "w", encoding="utf-8") as f:
            json.dump(aids, f, ensure_ascii=False, indent=2)

        logger.info(
            f"[INDEX] FAISS index built successfully with {index.ntotal} vectors"
        )
        logger.info(f"[INDEX] Index saved to: {config.FAISS_INDEX_PATH}")
        logger.info(
            f"[INDEX] Index-to-AID mapping saved to: {config.INDEX_TO_AID_PATH}"
        )

        return True

    except Exception as e:
        logger.error(f"[INDEX] Error building FAISS index: {e}", exc_info=True)
        return False


def train_cross_encoder_optimized(cross_encoder_data):
    """Huan luyen Cross-Encoder toi uu (DA SUA LOI num_samples=0)"""
    logger.info("[TRAIN] Training Cross-Encoder...")

    if cross_encoder_data is None or len(cross_encoder_data) < 10:
        logger.error(
            f"Cross-Encoder data is invalid or too small: {len(cross_encoder_data) if cross_encoder_data else 0} samples"
        )
        return None

    logger.info(
        f"[TRAIN] Starting Cross-Encoder training with {len(cross_encoder_data)} samples"
    )

    # --- BAT DAU PHAN SUA LOI ---
    # Chuyển đổi dữ liệu một cách an toàn
    dataset_dict = {"text1": [], "text2": [], "label": []}
    skipped_count = 0
    for i, pair in enumerate(cross_encoder_data):
        try:
            if "texts" in pair and len(pair["texts"]) == 2 and "label" in pair:
                text1 = str(pair["texts"][0] or "")
                text2 = str(pair["texts"][1] or "")
                if text1.strip() and text2.strip():
                    dataset_dict["text1"].append(text1)
                    dataset_dict["text2"].append(text2)
                    dataset_dict["label"].append(int(pair["label"]))
                else:
                    skipped_count += 1
            else:
                skipped_count += 1
        except Exception:
            skipped_count += 1

    if skipped_count > 0:
        logger.warning(
            f"[DATA] Skipped {skipped_count} invalid pairs during conversion."
        )

    if not dataset_dict["text1"]:
        logger.error(
            "[ERROR] No valid data remains after filtering for Cross-Encoder training."
        )
        return None
    # --- KET THUC PHAN SUA LOI ---

    # Create dataset
    dataset = Dataset.from_dict(dataset_dict)
    dataset_splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_splits["train"]
    eval_dataset = dataset_splits["test"]

    # Initialize model using config - Use PhoBERT-Law if available and contains model files
    phobert_law_path = config.PHOBERT_LAW_PATH
    if phobert_law_path.exists() and any(phobert_law_path.iterdir()):
        required_files = ["config.json", "pytorch_model.bin", "vocab.txt"]
        has_required_files = all(
            (phobert_law_path / file).exists() for file in required_files
        )
        if has_required_files:
            logger.info("[MODEL] Using PhoBERT-Law model (domain-adapted)")
            model_name = str(phobert_law_path)
        else:
            logger.warning(
                f"[MODEL] PhoBERT-Law directory exists but missing required files: {required_files}"
            )
            logger.info("[MODEL] Falling back to base PhoBERT model")
            model_name = config.CROSS_ENCODER_MODEL_NAME
    else:
        logger.info("[MODEL] Using base PhoBERT model")
        model_name = config.CROSS_ENCODER_MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # --- BAT DAU PHAN SUA LOI ---
    # Định nghĩa hàm tiền xử lý trực tiếp và đảm bảo nó hoạt động đúng
    def preprocess_function(examples):
        # Tokenizer nhận vào một danh sách các câu đầu tiên và một danh sách các câu thứ hai
        return tokenizer(
            examples["text1"],
            examples["text2"],
            truncation=True,
            padding="max_length",
            max_length=config.CROSS_ENCODER_MAX_LENGTH,
        )

    # --- KET THUC PHAN SUA LOI ---

    # Tokenize datasets
    logger.info("[TOKENIZE] Tokenizing training dataset...")
    train_dataset = train_dataset.map(preprocess_function, batched=True)

    logger.info("[TOKENIZE] Tokenizing evaluation dataset...")
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    # OPTIMIZED TRAINING ARGUMENTS with config parameters
    training_args = TrainingArguments(
        output_dir=str(config.MODELS_DIR / "cross_encoder_optimized"),
        num_train_epochs=config.CROSS_ENCODER_EPOCHS,
        per_device_train_batch_size=config.CROSS_ENCODER_BATCH_SIZE,
        per_device_eval_batch_size=config.CROSS_ENCODER_BATCH_SIZE,
        gradient_accumulation_steps=config.CROSS_ENCODER_GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.CROSS_ENCODER_LR,
        warmup_steps=config.CROSS_ENCODER_WARMUP_STEPS,
        weight_decay=0.01,
        logging_dir=str(config.LOGS_DIR),
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=config.CROSS_ENCODER_EVAL_STEPS,
        save_steps=config.CROSS_ENCODER_EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=config.CROSS_ENCODER_DATALOADER_PIN_MEMORY,
        dataloader_num_workers=(
            0 if os.name == "nt" else config.CROSS_ENCODER_DATALOADER_NUM_WORKERS
        ),
        dataloader_prefetch_factor=(
            None if os.name == "nt" else config.CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR
        ),
        fp16=config.FP16_TRAINING and torch.cuda.is_available(),
        save_total_limit=3,
        eval_accumulation_steps=2,
        lr_scheduler_type="linear",
        remove_unused_columns=True,
        report_to=None,
        dataloader_drop_last=True,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("[TRAIN] Starting Cross-Encoder training...")
    try:
        trainer.train()
        logger.info("[TRAIN] Training completed successfully!")
    except Exception as e:
        logger.error(f"[TRAIN] Training failed: {e}", exc_info=True)
        try:
            trainer.save_model()
            logger.info("[SAVE] Partial model saved despite training error")
        except Exception as save_error:
            logger.error(f"[SAVE] Failed to save partial model: {save_error}")
        return False

    try:
        trainer.save_model()
        logger.info(
            f"[SAVE] Cross-Encoder saved to: {config.MODELS_DIR / 'cross_encoder_optimized'}"
        )
    except Exception as e:
        logger.error(f"[SAVE] Failed to save model: {e}")
        return False

    del trainer, train_dataset, eval_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True


def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = (preds == labels).astype(float).mean()
    return {"accuracy": accuracy}


def run_evaluation_optimized():
    """Chạy evaluation toàn diện bằng cách sử dụng pipeline đã huấn luyện."""
    logger.info("[EVAL] Running comprehensive evaluation...")
    try:
        logger.info("[EVAL] Loading the newly trained pipeline...")
        pipeline = LegalQAPipeline(use_ensemble=True)
        if not pipeline.is_ready:
            logger.warning("[EVAL] Pipeline is not ready. Running basic evaluation.")
            return run_basic_evaluation()

        logger.info(
            f"[EVAL] Loading validation data from: {config.VAL_SPLIT_JSON_PATH}"
        )
        with open(config.VAL_SPLIT_JSON_PATH, "r", encoding="utf-8") as f:
            val_data = json.load(f)

        queries = [item["question"] for item in val_data]
        ground_truth_sets = [set(item["relevant_aids"]) for item in val_data]

        logger.info(f"[EVAL] Predicting on {len(queries)} validation queries...")
        retrieved_aids_batch = []
        per_query_results = []
        for i, query in enumerate(queries):
            try:
                results = pipeline.predict(
                    query=query, top_k_retrieval=config.TOP_K_RETRIEVAL, top_k_final=10
                )
                retrieved_aids = [res["aid"] for res in results]
                retrieved_aids_batch.append(retrieved_aids)
                per_query_results.append(
                    {
                        "qid": val_data[i].get("qid", f"query_{i}"),
                        "query": query,
                        "ground_truth_aids": list(ground_truth_sets[i]),
                        "predicted_aids": retrieved_aids,
                        "top_result_score": (
                            results[0]["rerank_score"] if results else 0
                        ),
                    }
                )
            except Exception as e:
                logger.warning(f"[EVAL] Error processing query {i}: {e}")
                retrieved_aids_batch.append([])
                per_query_results.append(
                    {
                        "qid": val_data[i].get("qid", f"query_{i}"),
                        "query": query,
                        "ground_truth_aids": list(ground_truth_sets[i]),
                        "predicted_aids": [],
                        "top_result_score": 0,
                    }
                )

        logger.info("[EVAL] Calculating metrics using BatchEvaluator...")
        evaluator = BatchEvaluator(k_values=[1, 3, 5, 10])
        reranking_metrics = evaluator.evaluate_batch(
            queries, ground_truth_sets, retrieved_aids_batch
        )

        logger.info("[EVAL] Creating comprehensive report...")
        reporter = EvaluationReporter()
        from datetime import datetime

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "bi_encoder_model": config.BI_ENCODER_MODEL_NAME,
            "cross_encoder_model": config.CROSS_ENCODER_MODEL_NAME,
            "dapt_model_used": (
                "PhoBERT-Law"
                if (config.PHOBERT_LAW_PATH / "config.json").exists()
                else "Base PhoBERT"
            ),
            "training_samples": len(queries),
        }
        report = reporter.create_comprehensive_report(
            retrieval_metrics={},
            reranking_metrics=reranking_metrics,
            per_query_results=per_query_results,
            metadata=metadata,
        )
        reporter.display_summary(report)
        report_path = reporter.save_report(report)
        logger.info(f"[EVAL] Evaluation report saved to {report_path}")
        return True
    except Exception as e:
        logger.error(f"[EVAL] An error occurred during evaluation: {e}", exc_info=True)
        return False


def run_basic_evaluation():
    """Chạy evaluation cơ bản khi pipeline không sẵn sàng."""
    logger.info("[EVAL] Running basic evaluation as fallback...")
    try:
        basic_metrics = {
            "precision@1": 0.0,
            "recall@1": 0.0,
            "f1@1": 0.0,
            "precision@5": 0.0,
            "recall@5": 0.0,
            "f1@5": 0.0,
        }
        from datetime import datetime

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "basic_fallback",
            "note": "Pipeline not ready",
        }
        reporter = EvaluationReporter()
        report = reporter.create_comprehensive_report(
            retrieval_metrics=basic_metrics,
            reranking_metrics=basic_metrics,
            per_query_results=[],
            metadata=metadata,
        )
        reporter.display_summary(report)
        report_path = reporter.save_report(report)
        logger.info(f"[EVAL] Basic evaluation report saved to {report_path}")
        return True
    except Exception as e:
        logger.error(f"[EVAL] Error in basic evaluation: {e}")
        return False


def train_light_reranker_optimized(cross_encoder_data):
    """Train Light Reranker model for cascaded reranking"""
    logger.info("[LIGHT-RERANKER] Starting Light Reranker training...")
    try:
        if cross_encoder_data is None or len(cross_encoder_data) < 10:
            logger.error(f"Light Reranker data is invalid or too small.")
            return False

        dataset_dict = {"text1": [], "text2": [], "label": []}
        for pair in cross_encoder_data:
            if "texts" in pair and len(pair["texts"]) == 2 and "label" in pair:
                dataset_dict["text1"].append(str(pair["texts"][0] or ""))
                dataset_dict["text2"].append(str(pair["texts"][1] or ""))
                dataset_dict["label"].append(int(pair["label"]))

        dataset = Dataset.from_dict(dataset_dict)
        dataset_splits = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_splits["train"]
        eval_dataset = dataset_splits["test"]

        model_name = config.LIGHT_RERANKER_MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def preprocess_function(examples):
            return tokenizer(
                examples["text1"],
                examples["text2"],
                truncation=True,
                padding="max_length",
                max_length=config.LIGHT_RERANKER_MAX_LENGTH,
            )

        train_dataset = train_dataset.map(preprocess_function, batched=True)
        eval_dataset = eval_dataset.map(preprocess_function, batched=True)

        training_args = TrainingArguments(
            output_dir=str(config.LIGHT_RERANKER_PATH),
            overwrite_output_dir=True,
            num_train_epochs=2,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir=str(config.LIGHT_RERANKER_PATH / "logs"),
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=400,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=config.FP16_TRAINING and torch.cuda.is_available(),
            dataloader_pin_memory=config.CROSS_ENCODER_DATALOADER_PIN_MEMORY,
            dataloader_num_workers=(
                0 if os.name == "nt" else config.CROSS_ENCODER_DATALOADER_NUM_WORKERS
            ),
            report_to=None,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(config.LIGHT_RERANKER_PATH)
        logger.info(f"[LIGHT-RERANKER] Model saved to: {config.LIGHT_RERANKER_PATH}")
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    except Exception as e:
        logger.error(
            f"[LIGHT-RERANKER] Error during Light Reranker training: {e}", exc_info=True
        )
        return False


def run_complete_training_pipeline():
    """Chay pipeline hoan chinh voi CHECKPOINTING"""
    logger.info("=" * 60)
    logger.info("MODEL TRAINING & EVALUATION PIPELINE (with Checkpointing)")
    logger.info("=" * 60)

    # Tải checkpoint
    checkpoint_state = load_checkpoint()

    try:
        # Tải dữ liệu một lần
        logger.info("PRE-STEP: Loading all prepared training data...")
        bi_encoder_data, cross_encoder_data = load_prepared_training_data()
        if bi_encoder_data is None or cross_encoder_data is None:
            logger.error("Failed to load initial training data. Aborting pipeline.")
            return False

        # Step 1: Train Bi-Encoder
        if not is_step_complete(checkpoint_state, "train_bi_encoder"):
            logger.info("STEP 1: Training Bi-Encoder...")
            bi_encoder_model = train_bi_encoder_optimized(bi_encoder_data)
            if bi_encoder_model is None:
                logger.error("Bi-Encoder training failed")
                return False
            mark_step_complete(checkpoint_state, "train_bi_encoder")
        else:
            logger.info("STEP 1: Training Bi-Encoder... [SKIPPED - Already complete]")
            # Tải lại model đã train để dùng cho bước sau
            bi_encoder_model = SentenceTransformer(
                str(config.MODELS_DIR / "bi_encoder_optimized")
            )

        # Step 2: Build FAISS index
        if not is_step_complete(checkpoint_state, "build_faiss_index"):
            logger.info("STEP 2: Building FAISS index...")
            if not build_faiss_index_optimized(bi_encoder_model):
                logger.error("FAISS index building failed")
                return False
            mark_step_complete(checkpoint_state, "build_faiss_index")
        else:
            logger.info("STEP 2: Building FAISS index... [SKIPPED - Already complete]")

        # Giải phóng bộ nhớ của Bi-Encoder model
        del bi_encoder_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 3: Train Cross-Encoder
        if not is_step_complete(checkpoint_state, "train_cross_encoder"):
            logger.info("STEP 3: Training Cross-Encoder...")
            if not train_cross_encoder_optimized(cross_encoder_data):
                logger.error("Cross-Encoder training failed")
                return False
            mark_step_complete(checkpoint_state, "train_cross_encoder")
        else:
            logger.info(
                "STEP 3: Training Cross-Encoder... [SKIPPED - Already complete]"
            )

        # Step 4: Train Light Reranker
        if not is_step_complete(checkpoint_state, "train_light_reranker"):
            logger.info("STEP 4: Training Light Reranker...")
            if not train_light_reranker_optimized(cross_encoder_data):
                logger.warning("Light Reranker training failed, but continuing...")
            mark_step_complete(checkpoint_state, "train_light_reranker")
        else:
            logger.info(
                "STEP 4: Training Light Reranker... [SKIPPED - Already complete]"
            )

        # Step 5: Run evaluation
        if not is_step_complete(checkpoint_state, "run_evaluation"):
            logger.info("STEP 5: Running evaluation...")
            if not run_evaluation_optimized():
                logger.warning("Evaluation failed, but pipeline completed")
            mark_step_complete(checkpoint_state, "run_evaluation")
        else:
            logger.info("STEP 5: Running evaluation... [SKIPPED - Already complete]")

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"Error during pipeline: {e}", exc_info=True)
        return False


def main():
    """Main function"""
    success = run_complete_training_pipeline()

    if success:
        logger.info("✅ Model Training & Evaluation Pipeline completed successfully!")
    else:
        logger.error("❌ Model Training & Evaluation Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
