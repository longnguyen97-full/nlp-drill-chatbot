#!/usr/bin/env python3
"""
Model Training & Evaluation Pipeline - Script Toi Uu Can Bang
============================================================

Script nay huan luyen Bi-Encoder, build FAISS index, huan luyen Cross-Encoder,
va evaluation trong mot buoc toi uu can bang giua hieu qua va de hieu.
PHAN MEM DA TICH HOP CHECKPOINT DE CO THE KHOI DONG LAI.

Tac gia: LawBot Team
Phien ban: Balanced Optimized Pipeline v8.0 (Refactored & Optimized)
"""

import json
import logging
import torch
import faiss
import numpy as np
import random
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
from datetime import datetime

# --- System Path Setup ---
sys.path.append(str(Path(__file__).parent.parent))
import config
from core.logging_system import get_logger
from core.pipeline import LegalQAPipeline
from core.evaluation_reporter import BatchEvaluator, EvaluationReporter

# --- Global Logger ---
logger = get_logger(__name__)

# --- Checkpointing Constants & Functions ---
CHECKPOINT_FILE = config.DATA_PROCESSED_DIR / "pipeline_checkpoint.json"


def load_checkpoint():
    """Tải trạng thái pipeline từ file checkpoint."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                logger.info(f"[CHECKPOINT] Found checkpoint file at {CHECKPOINT_FILE}")
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(
                f"[CHECKPOINT] Could not read checkpoint file: {e}. Starting fresh."
            )
    return {}


def save_checkpoint(state):
    """Lưu trạng thái pipeline vào file checkpoint."""
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        logger.info(f"[CHECKPOINT] Saved checkpoint: {state}")
    except IOError as e:
        logger.error(f"[CHECKPOINT] Could not save checkpoint file: {e}")


def mark_step_complete(state, step_name):
    """Đánh dấu một bước đã hoàn thành và lưu checkpoint."""
    state[step_name] = True
    save_checkpoint(state)


def is_step_complete(state, step_name):
    """Kiểm tra xem một bước đã hoàn thành chưa."""
    return state.get(step_name, False)


# --- Data Loading and Preparation ---


def load_jsonl_data(file_path, model_name):
    """Hàm chung để tải dữ liệu từ file .jsonl với error handling."""
    if not file_path.exists():
        logger.error(
            f"[{model_name}] Training data not found at {file_path}. Please run data preparation first."
        )
        return None

    data_list = []
    errors = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                data_list.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.warning(f"[{model_name}] Line {i}: JSON decode error - {e}")
                errors += 1

    logger.info(
        f"[{model_name}] Loaded {len(data_list)} samples with {errors} errors from {file_path.name}"
    )
    return data_list if data_list else None


def create_bi_encoder_examples(triplets, data_type="Training"):
    """Tạo InputExample cho Bi-Encoder từ dữ liệu triplets."""
    logger.info(f"[BI-ENCODER] Creating {data_type} examples...")
    examples, skipped = [], 0
    for i, triplet in enumerate(triplets):
        anchor = str(triplet.get("anchor", ""))
        positive = str(triplet.get("positive", ""))
        negative = str(triplet.get("negative", ""))

        if not all((anchor.strip(), positive.strip(), negative.strip())):
            skipped += 1
            continue

        examples.append(InputExample(texts=[anchor, positive], label=1.0))
        examples.append(InputExample(texts=[anchor, negative], label=0.0))

    logger.info(
        f"[BI-ENCODER] Created {len(examples)} {data_type} examples, skipped {skipped} invalid triplets."
    )
    return examples


# --- Core Training and Indexing Functions ---


def train_bi_encoder_optimized(bi_encoder_data):
    """Huấn luyện Bi-Encoder với validation và tối ưu hóa."""
    logger.info("[BI-ENCODER] Starting Bi-Encoder training process...")

    if not bi_encoder_data or len(bi_encoder_data) < 10:
        logger.error(
            f"Not enough data for Bi-Encoder training: {len(bi_encoder_data) if bi_encoder_data else 0} samples."
        )
        return None

    random.shuffle(bi_encoder_data)
    train_size = int(len(bi_encoder_data) * 0.9)
    train_triplets, val_triplets = (
        bi_encoder_data[:train_size],
        bi_encoder_data[train_size:],
    )

    train_examples = create_bi_encoder_examples(train_triplets, "Training")
    val_examples = create_bi_encoder_examples(val_triplets, "Validation")

    if not train_examples:
        logger.error("No valid training examples for Bi-Encoder. Aborting.")
        return None

    try:
        model = SentenceTransformer(config.BI_ENCODER_MODEL_NAME)
        train_loss = losses.ContrastiveLoss(model)

        # Adaptive batch sizing based on memory
        import psutil

        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)

        # Adaptive batch size
        if available_memory_gb >= 8:
            batch_size = config.BI_ENCODER_BATCH_SIZE
        elif available_memory_gb >= 4:
            batch_size = max(16, config.BI_ENCODER_BATCH_SIZE // 2)
        else:
            batch_size = max(8, config.BI_ENCODER_BATCH_SIZE // 4)

        logger.info(
            f"[BI-ENCODER] Available memory: {available_memory_gb:.1f} GB, using batch size: {batch_size}"
        )

        num_workers = 0 if os.name == "nt" else config.BI_ENCODER_DATALOADER_NUM_WORKERS
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        evaluator = (
            EmbeddingSimilarityEvaluator.from_input_examples(
                val_examples, name="bi-val"
            )
            if val_examples
            else None
        )

        # Note: EmbeddingSimilarityEvaluator, SentenceTransformer, InputExample, losses already imported at top

        # Memory monitoring
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"[BI-ENCODER] Initial GPU memory: {initial_memory:.2f} GB")

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=5,  # Increased for better learning
            warmup_steps=100,  # Use steps for compatibility
            optimizer_params={
                "lr": 2e-5,  # Optimized learning rate
                "eps": 1e-6,
            },
            evaluator=evaluator,
            output_path=str(config.BI_ENCODER_PATH),
            # Additional optimizations
            show_progress_bar=True,
        )

        # Memory cleanup
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"[BI-ENCODER] Final GPU memory: {final_memory:.2f} GB")
            torch.cuda.empty_cache()

        model.save(str(config.BI_ENCODER_PATH))
        logger.info(
            f"[BI-ENCODER] Training complete. Model saved to: {config.BI_ENCODER_PATH}"
        )
        return model

    except Exception as e:
        logger.error(f"[BI-ENCODER] Training failed: {e}", exc_info=True)
        # Cleanup on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None


def build_faiss_index_optimized(model):
    """Xây dựng FAISS index từ model Bi-Encoder đã huấn luyện."""
    logger.info("[FAISS] Building FAISS index...")
    try:
        from core.utils import parse_legal_corpus

        all_articles = parse_legal_corpus(config.LEGAL_CORPUS_PATH)
        if not all_articles:
            logger.error("[FAISS] No articles found in legal corpus. Aborting.")
            return False

        documents = [article["content"] for article in all_articles]
        aids = [article["aid"] for article in all_articles]

        logger.info(f"[FAISS] Encoding {len(documents)} documents...")
        embeddings = model.encode(
            documents,
            batch_size=config.BI_ENCODER_BATCH_SIZE
            * 2,  # Increase batch size for inference
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype(np.float32))

        config.INDEXES_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(config.FAISS_INDEX_PATH))
        with open(config.INDEX_TO_AID_PATH, "w", encoding="utf-8") as f:
            json.dump(aids, f)

        logger.info(
            f"[FAISS] Index with {index.ntotal} vectors built and saved successfully."
        )
        return True

    except Exception as e:
        logger.error(f"[FAISS] Index building failed: {e}", exc_info=True)
        return False


def _prepare_reranker_data(raw_data, model_name):
    """Hàm chung để chuẩn bị dữ liệu cho các mô hình Reranker."""
    dataset_dict = {"text1": [], "text2": [], "label": []}
    skipped, invalid_labels = 0, 0
    for pair in raw_data:
        texts = pair.get("texts")
        label = pair.get("label")
        if isinstance(texts, list) and len(texts) == 2 and label in [0, 1]:
            text1, text2 = str(texts[0] or ""), str(texts[1] or "")
            if text1.strip() and text2.strip():
                dataset_dict["text1"].append(text1)
                dataset_dict["text2"].append(text2)
                dataset_dict["label"].append(int(label))
            else:
                skipped += 1
        else:
            if label not in [0, 1]:
                invalid_labels += 1
            skipped += 1

    logger.info(
        f"[{model_name}] Prepared {len(dataset_dict['text1'])} valid pairs. Skipped: {skipped}, Invalid Labels: {invalid_labels}."
    )
    return Dataset.from_dict(dataset_dict) if dataset_dict["text1"] else None


def _train_reranker(
    model_name_or_path, training_data, training_args, max_length, model_log_name
):
    """Hàm chung để huấn luyện các mô hình Reranker."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=2
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"[{model_log_name}] Tokenizer pad_token set to eos_token.")

        if len(tokenizer) != model.config.vocab_size:
            logger.warning(
                f"[{model_log_name}] Vocab size mismatch. Resizing model embeddings to {len(tokenizer)}."
            )
            model.resize_token_embeddings(len(tokenizer))

        def preprocess_function(examples):
            # Enhanced data cleaning and validation
            try:
                cleaned_text1 = []
                cleaned_text2 = []

                for text1, text2 in zip(examples["text1"], examples["text2"]):
                    # Enhanced cleaning with more aggressive filtering
                    import re
                    
                    # Remove ALL problematic characters including unicode
                    text1_clean = re.sub(r'[^\w\s\u00C0-\u1EF9]+', ' ', str(text1)).strip()
                    text2_clean = re.sub(r'[^\w\s\u00C0-\u1EF9]+', ' ', str(text2)).strip()
                    
                    # Remove extra whitespace
                    text1_clean = re.sub(r'\s+', ' ', text1_clean).strip()
                    text2_clean = re.sub(r'\s+', ' ', text2_clean).strip()
                    
                    # Ensure minimum and maximum length
                    if not text1_clean or len(text1_clean) < 3:
                        text1_clean = "text"
                    if not text2_clean or len(text2_clean) < 3:
                        text2_clean = "text"
                    
                    # Strict length limits
                    text1_clean = text1_clean[:max_length//2]
                    text2_clean = text2_clean[:max_length//2]
                    
                    cleaned_text1.append(text1_clean)
                    cleaned_text2.append(text2_clean)

                # Tokenize with strict error handling
                result = tokenizer(
                    cleaned_text1,
                    cleaned_text2,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors=None,
                )

                # Strict token ID validation
                vocab_size = len(tokenizer)
                unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
                
                for key in ["input_ids", "attention_mask"]:
                    if key in result:
                        for i, ids in enumerate(result[key]):
                            # Replace ALL invalid tokens with UNK
                            result[key][i] = [
                                unk_id if token_id >= vocab_size or token_id < 0 else token_id
                                for token_id in ids
                            ]

                return result
            except Exception as e:
                logger.error(f"[{model_log_name}] Critical preprocessing error: {e}")
                # Return safe fallback
                return {
                    "input_ids": [[tokenizer.cls_token_id] + [unk_id] * (max_length-2) + [tokenizer.sep_token_id]],
                    "attention_mask": [[1] * max_length],
                    "labels": [0]
                }

        dataset_splits = training_data.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_splits["train"].map(preprocess_function, batched=True)
        eval_dataset = dataset_splits["test"].map(preprocess_function, batched=True)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=lambda p: {
                "accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()
            },
        )

        # Set environment variables to help with CUDA debugging
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"

        # Memory cleanup before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(
                f"[{model_log_name}] GPU memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )

        try:
            trainer.train()
        except RuntimeError as e:
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                logger.error(f"[{model_log_name}] CUDA/Memory error during training: {e}")
                
                # Try with reduced batch size first
                logger.info(f"[{model_log_name}] Attempting with reduced batch size...")
                training_args.per_device_train_batch_size = max(1, training_args.per_device_train_batch_size // 2)
                training_args.per_device_eval_batch_size = max(1, training_args.per_device_eval_batch_size // 2)
                
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    tokenizer=tokenizer,
                )
                
                try:
                    trainer.train()
                except RuntimeError as e2:
                    logger.error(f"[{model_log_name}] Still failing with reduced batch size: {e2}")
                    # Final fallback to CPU
                    logger.info(f"[{model_log_name}] Attempting CPU training...")
                    training_args.device = torch.device("cpu")
                    training_args.per_device_train_batch_size = 4
                    training_args.per_device_eval_batch_size = 4
                    
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        tokenizer=tokenizer,
                    )
                    trainer.train()
            else:
                raise
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)

        logger.info(
            f"[{model_log_name}] Training complete. Model saved to {training_args.output_dir}"
        )
        del trainer, model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    except Exception as e:
        logger.error(f"[{model_log_name}] Training failed: {e}", exc_info=True)
        return False


def run_comprehensive_evaluation():
    """Chạy đánh giá toàn diện với đầy đủ metrics và báo cáo chi tiết."""
    logger.info("[EVAL] Starting comprehensive evaluation...")
    try:
        # 1. Khởi tạo pipeline
        logger.info("[EVAL] Initializing pipeline...")
        pipeline = LegalQAPipeline(use_ensemble=True)
        if not pipeline.is_ready:
            logger.error("[EVAL] Pipeline is not ready. Cannot run evaluation.")
            return False

        # 2. Load validation data
        logger.info("[EVAL] Loading validation data...")
        if not config.VAL_SPLIT_JSON_PATH.exists():
            logger.error(
                f"[EVAL] Validation data not found at {config.VAL_SPLIT_JSON_PATH}"
            )
            return False

        with open(config.VAL_SPLIT_JSON_PATH, "r", encoding="utf-8") as f:
            val_data = json.load(f)

        queries = [item["question"] for item in val_data]
        ground_truth_sets = [set(item["relevant_aids"]) for item in val_data]

        logger.info(f"[EVAL] Loaded {len(queries)} validation queries")

        # 3. Đánh giá retrieval (tầng 1 - Bi-Encoder)
        logger.info("[EVAL] Evaluating retrieval performance (Tier 1 - Bi-Encoder)...")
        retrieval_predictions = []
        retrieval_scores = []

        for i, q in enumerate(queries):
            if i % 10 == 0:
                logger.info(f"[EVAL] Processing retrieval query {i+1}/{len(queries)}")
            try:
                retrieved_aids, distances = pipeline.retrieve(q, config.TOP_K_RETRIEVAL)
                retrieval_predictions.append(retrieved_aids)
                # Convert distances to similarity scores (1 - normalized_distance)
                max_dist = max(distances) if distances else 1.0
                scores = [
                    1.0 - (d / max_dist) if max_dist > 0 else 0.0 for d in distances
                ]
                retrieval_scores.append(scores)
            except Exception as e:
                logger.error(f"[EVAL] Error in retrieval for query {i}: {e}")
                retrieval_predictions.append([])
                retrieval_scores.append([])

        # 4. Đánh giá reranking (tầng 3 - Cross-Encoder)
        logger.info(
            "[EVAL] Evaluating reranking performance (Tier 3 - Cross-Encoder)..."
        )
        reranking_predictions = []
        reranking_scores = []

        for i, q in enumerate(queries):
            if i % 10 == 0:
                logger.info(f"[EVAL] Processing reranking query {i+1}/{len(queries)}")
            try:
                results = pipeline.predict(
                    q, top_k_retrieval=config.TOP_K_RETRIEVAL, top_k_final=10
                )
                reranking_predictions.append(results)
                scores = [res.get("rerank_score", 0.0) for res in results]
                reranking_scores.append(scores)
            except Exception as e:
                logger.error(f"[EVAL] Error in reranking for query {i}: {e}")
                reranking_predictions.append([])
                reranking_scores.append([])

        # 5. Tính toán metrics chi tiết
        logger.info("[EVAL] Computing comprehensive metrics...")

        # Retrieval metrics
        evaluator = BatchEvaluator(k_values=[1, 3, 5, 10, 20, 50])
        retrieval_aids_batch = [
            [aid for aid in preds] for preds in retrieval_predictions
        ]
        retrieval_metrics = evaluator.evaluate_batch(
            queries, ground_truth_sets, retrieval_aids_batch
        )

        # Reranking metrics
        reranking_aids_batch = [
            [res["aid"] for res in preds] for preds in reranking_predictions
        ]
        reranking_metrics = evaluator.evaluate_batch(
            queries, ground_truth_sets, reranking_aids_batch
        )

        # 6. Tạo per-query results chi tiết
        logger.info("[EVAL] Creating detailed per-query results...")
        per_query_results = []

        for i, (
            query,
            gt_set,
            ret_aids,
            ret_scores,
            rerank_results,
            rerank_scores,
        ) in enumerate(
            zip(
                queries,
                ground_truth_sets,
                retrieval_predictions,
                retrieval_scores,
                reranking_predictions,
                reranking_scores,
            )
        ):
            # Tính precision, recall, F1 cho retrieval
            ret_precision = (
                len(set(ret_aids) & gt_set) / len(ret_aids) if ret_aids else 0.0
            )
            ret_recall = len(set(ret_aids) & gt_set) / len(gt_set) if gt_set else 0.0
            ret_f1 = (
                2 * (ret_precision * ret_recall) / (ret_precision + ret_recall)
                if (ret_precision + ret_recall) > 0
                else 0.0
            )

            # Tính precision, recall, F1 cho reranking
            rerank_aids = [res["aid"] for res in rerank_results]
            rerank_precision = (
                len(set(rerank_aids) & gt_set) / len(rerank_aids)
                if rerank_aids
                else 0.0
            )
            rerank_recall = (
                len(set(rerank_aids) & gt_set) / len(gt_set) if gt_set else 0.0
            )
            rerank_f1 = (
                2
                * (rerank_precision * rerank_recall)
                / (rerank_precision + rerank_recall)
                if (rerank_precision + rerank_recall) > 0
                else 0.0
            )

            per_query_results.append(
                {
                    "query_id": i,
                    "query": query,
                    "ground_truth": list(gt_set),
                    "ground_truth_count": len(gt_set),
                    "retrieval_results": {
                        "aids": ret_aids[:10],
                        "scores": ret_scores[:10],
                        "precision": ret_precision,
                        "recall": ret_recall,
                        "f1": ret_f1,
                        "found_relevant": len(set(ret_aids) & gt_set),
                    },
                    "reranking_results": {
                        "aids": rerank_aids,
                        "scores": rerank_scores,
                        "precision": rerank_precision,
                        "recall": rerank_recall,
                        "f1": rerank_f1,
                        "found_relevant": len(set(rerank_aids) & gt_set),
                    },
                    "improvement": {
                        "precision_improvement": rerank_precision - ret_precision,
                        "recall_improvement": rerank_recall - ret_recall,
                        "f1_improvement": rerank_f1 - ret_f1,
                    },
                }
            )

        # 7. Tạo metadata chi tiết
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "comprehensive",
            "total_queries": len(queries),
            "pipeline_config": {
                "top_k_retrieval": config.TOP_K_RETRIEVAL,
                "top_k_final": 10,
                "use_ensemble": True,
                "use_cascaded_reranking": True,
            },
            "model_paths": {
                "bi_encoder": str(config.BI_ENCODER_PATH),
                "cross_encoder": str(config.CROSS_ENCODER_PATH),
                "light_reranker": str(config.LIGHT_RERANKER_PATH),
                "faiss_index": str(config.FAISS_INDEX_PATH),
            },
            "summary_stats": {
                "avg_ground_truth_per_query": sum(len(gt) for gt in ground_truth_sets)
                / len(ground_truth_sets),
                "queries_with_ground_truth": len(
                    [gt for gt in ground_truth_sets if len(gt) > 0]
                ),
                "avg_retrieval_candidates": sum(
                    len(ret) for ret in retrieval_predictions
                )
                / len(retrieval_predictions),
                "avg_reranking_candidates": sum(
                    len(rerank) for rerank in reranking_predictions
                )
                / len(reranking_predictions),
            },
        }

        # 8. Tạo và lưu báo cáo toàn diện
        logger.info("[EVAL] Generating comprehensive report...")
        reporter = EvaluationReporter()
        report = reporter.create_comprehensive_report(
            retrieval_metrics=retrieval_metrics,
            reranking_metrics=reranking_metrics,
            per_query_results=per_query_results,
            metadata=metadata,
        )

        # 9. Hiển thị và lưu báo cáo
        reporter.display_summary(report)
        report_path = reporter.save_report(report)

        # 10. Hiển thị tóm tắt chi tiết
        logger.info("=" * 80)
        logger.info("📊 COMPREHENSIVE EVALUATION RESULTS:")
        logger.info("=" * 80)

        # Retrieval summary
        logger.info("🎯 RETRIEVAL PERFORMANCE (Tier 1 - Bi-Encoder):")
        for k, metrics in retrieval_metrics.items():
            logger.info(
                f"  Top-{k}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}"
            )

        # Reranking summary
        logger.info("⚡ RERANKING PERFORMANCE (Tier 3 - Cross-Encoder):")
        for k, metrics in reranking_metrics.items():
            logger.info(
                f"  Top-{k}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}"
            )

        # Improvement summary
        logger.info("📈 PERFORMANCE IMPROVEMENT (Reranking vs Retrieval):")
        avg_improvements = {
            "precision": sum(
                r["improvement"]["precision_improvement"] for r in per_query_results
            )
            / len(per_query_results),
            "recall": sum(
                r["improvement"]["recall_improvement"] for r in per_query_results
            )
            / len(per_query_results),
            "f1": sum(r["improvement"]["f1_improvement"] for r in per_query_results)
            / len(per_query_results),
        }
        logger.info(
            f"  Average Precision Improvement: {avg_improvements['precision']:.4f}"
        )
        logger.info(f"  Average Recall Improvement: {avg_improvements['recall']:.4f}")
        logger.info(f"  Average F1 Improvement: {avg_improvements['f1']:.4f}")

        logger.info(f"📄 Detailed report saved to: {report_path}")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"[EVAL] Comprehensive evaluation failed: {e}", exc_info=True)
        return False


# --- Main Pipeline Execution ---


def main():
    """Hàm chính điều khiển toàn bộ pipeline huấn luyện và đánh giá."""
    logger.info("=" * 80)
    logger.info("STARTING: Model Training & Evaluation Pipeline v8.0")
    logger.info("=" * 80)
    logger.info("Pipeline Overview:")
    logger.info("   - Step 1: Bi-Encoder Training (Sentence Transformers)")
    logger.info("   - Step 2: FAISS Index Building (Vector Search)")
    logger.info("   - Step 3: Cross-Encoder Training (Sequence Classification)")
    logger.info("   - Step 4: Light Reranker Training (Fast Reranking)")
    logger.info("   - Step 5: Comprehensive Evaluation (Full Metrics)")
    logger.info("=" * 80)

    checkpoint_state = load_checkpoint()

    try:
        # --- Data Loading ---
        bi_encoder_data = load_jsonl_data(
            config.BI_ENCODER_TRAIN_MIXED_PATH, "Bi-Encoder"
        )
        reranker_data = load_jsonl_data(config.TRAIN_PAIRS_MIXED_PATH, "Reranker")
        if not bi_encoder_data or not reranker_data:
            raise RuntimeError("Failed to load necessary training data.")

        # --- Step 1: Bi-Encoder Training ---
        if not is_step_complete(checkpoint_state, "train_bi_encoder"):
            bi_encoder_model = train_bi_encoder_optimized(bi_encoder_data)
            if not bi_encoder_model:
                raise RuntimeError("Bi-Encoder training failed.")
            mark_step_complete(checkpoint_state, "train_bi_encoder")
        else:
            logger.info("STEP 1: Bi-Encoder Training... [SKIPPED - Already complete]")
            bi_encoder_model = SentenceTransformer(str(config.BI_ENCODER_PATH))

        # --- Step 2: FAISS Index Building ---
        if not is_step_complete(checkpoint_state, "build_faiss_index"):
            if not build_faiss_index_optimized(bi_encoder_model):
                raise RuntimeError("FAISS index building failed.")
            mark_step_complete(checkpoint_state, "build_faiss_index")
        else:
            logger.info("STEP 2: FAISS Index... [SKIPPED - Already complete]")

        del bi_encoder_model  # Free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Step 3: Cross-Encoder Training ---
        if not is_step_complete(checkpoint_state, "train_cross_encoder"):
            logger.info("STEP 3: Cross-Encoder Training...")
            dataset = _prepare_reranker_data(reranker_data, "Cross-Encoder")
            if dataset:
                args = TrainingArguments(
                    output_dir=str(config.CROSS_ENCODER_PATH),
                    num_train_epochs=config.CROSS_ENCODER_EPOCHS,
                    per_device_train_batch_size=config.CROSS_ENCODER_BATCH_SIZE,
                    learning_rate=config.CROSS_ENCODER_LR,
                    warmup_steps=config.CROSS_ENCODER_WARMUP_RATIO,
                    eval_steps=config.CROSS_ENCODER_EVAL_STEPS,
                    save_steps=config.CROSS_ENCODER_EVAL_STEPS * 2,
                    report_to="none",
                    # Memory optimizations
                    gradient_checkpointing=True,
                    dataloader_pin_memory=False,
                    remove_unused_columns=False,
                    # Additional optimizations
                    fp16=config.FP16_TRAINING,
                    gradient_accumulation_steps=config.CROSS_ENCODER_GRADIENT_ACCUMULATION_STEPS,
                    dataloader_num_workers=config.CROSS_ENCODER_DATALOADER_NUM_WORKERS,
                )
                if not _train_reranker(
                    config.CROSS_ENCODER_MODEL_NAME,
                    dataset,
                    args,
                    config.CROSS_ENCODER_MAX_LENGTH,
                    "Cross-Encoder",
                ):
                    logger.error("Cross-Encoder training failed.")
                    raise RuntimeError("Cross-Encoder training failed.")
                mark_step_complete(checkpoint_state, "train_cross_encoder")
            else:
                logger.error("Failed to prepare Cross-Encoder dataset.")
                raise RuntimeError("Cross-Encoder dataset preparation failed.")
        else:
            logger.info("STEP 3: Cross-Encoder Training... [SKIPPED - Already complete]")

        # --- Step 4: Light Reranker Training ---
        if not is_step_complete(checkpoint_state, "train_light_reranker"):
            logger.info("STEP 4: Light Reranker Training...")
            dataset = _prepare_reranker_data(reranker_data, "Light-Reranker")
            if dataset:
                args = TrainingArguments(
                    output_dir=str(config.LIGHT_RERANKER_PATH),
                    num_train_epochs=2,
                    per_device_train_batch_size=config.LIGHT_RERANKER_BATCH_SIZE,
                    learning_rate=config.CROSS_ENCODER_LR,
                    warmup_steps=50,
                    eval_steps=200,
                    save_steps=200,
                    report_to="none",
                    # Memory optimizations
                    gradient_checkpointing=True,
                    dataloader_pin_memory=False,
                    remove_unused_columns=False,
                )
                if not _train_reranker(
                    config.LIGHT_RERANKER_MODEL_NAME,
                    dataset,
                    args,
                    config.LIGHT_RERANKER_MAX_LENGTH,
                    "Light-Reranker",
                ):
                    logger.error("Light Reranker training failed.")
                    raise RuntimeError("Light Reranker training failed.")
                mark_step_complete(checkpoint_state, "train_light_reranker")
            else:
                logger.error("Failed to prepare Light Reranker dataset.")
                raise RuntimeError("Light Reranker dataset preparation failed.")
        else:
            logger.info("STEP 4: Light Reranker Training... [SKIPPED - Already complete]")

        # --- Step 5: Comprehensive Evaluation ---
        if not is_step_complete(checkpoint_state, "run_evaluation"):
            logger.info("STEP 5: Comprehensive Evaluation...")
            if not run_comprehensive_evaluation():
                logger.warning("Evaluation run failed, but training steps are complete.")
            mark_step_complete(checkpoint_state, "run_evaluation")
        else:
            logger.info("STEP 5: Comprehensive Evaluation... [SKIPPED - Already complete]")

    except Exception as e:
        logger.error(
            f"PIPELINE HALTED: An unrecoverable error occurred: {e}", exc_info=True
        )
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
