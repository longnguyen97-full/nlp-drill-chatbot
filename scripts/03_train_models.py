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
    EarlyStoppingCallback,
)
from datasets import Dataset
import sys
from datetime import datetime
import inspect

# --- Early CLI pre-parse to set performance mode before importing config anywhere ---
def _apply_mode_from_cli_early():
    try:
        import sys as _sys
        import os as _os
        selected_mode = None
        for i, arg in enumerate(_sys.argv[1:], start=1):
            if arg == "--mode" and i + 1 < len(_sys.argv):
                selected_mode = _sys.argv[i + 1]
                break
            if arg.startswith("--mode="):
                selected_mode = arg.split("=", 1)[1]
                break
        if selected_mode:
            selected_mode = selected_mode.strip().lower()
            if selected_mode in ("fast", "quality"):
                _os.environ["LAWBOT_PERFORMANCE_MODE"] = selected_mode
    except Exception:
        pass

_apply_mode_from_cli_early()

# --- System Path Setup ---
# Ensure we can import from project root regardless of how script is called
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set working directory to project root
os.chdir(project_root)

# --- Config Import with Performance Mode Support ---
def clear_config_cache():
    """Clear all config-related module cache to force reload"""
    modules_to_clear = ['config', 'config_fast', 'config_quality', 'config_base']
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]

def get_performance_mode():
    """Get performance mode from environment variable and force reload config if needed"""
    # Check environment variable first
    env_performance_mode = os.getenv("LAWBOT_PERFORMANCE_MODE", "quality")
    print(f"[CONFIG] Environment variable LAWBOT_PERFORMANCE_MODE={env_performance_mode}")
    
    try:
        # Clear config cache first
        clear_config_cache()
        
        # Import config
        import config
        
        # If environment variable doesn't match config, force reload again
        if env_performance_mode != config.PERFORMANCE_MODE:
            print(f"[CONFIG] Environment variable LAWBOT_PERFORMANCE_MODE={env_performance_mode} but config.PERFORMANCE_MODE={config.PERFORMANCE_MODE}")
            print(f"[CONFIG] Forcing config reload...")
            
            # Clear cache again and re-import
            clear_config_cache()
            import config
            print(f"[CONFIG] After force reload - PERFORMANCE_MODE: {config.PERFORMANCE_MODE}")
        
        return config.PERFORMANCE_MODE
    except ImportError:
        print(f"[CONFIG] Could not import config, using environment variable: {env_performance_mode}")
        return env_performance_mode

# Get current performance mode and import config
PERFORMANCE_MODE = get_performance_mode()
print(f"[CONFIG] Final PERFORMANCE_MODE: {PERFORMANCE_MODE}")

# Import config after ensuring correct mode
import config
import importlib
from core.logging_system import get_logger
from core.pipeline import LegalQAPipeline
from core.evaluation_reporter import BatchEvaluator, EvaluationReporter
from core.utils.aid_utils import canonicalize_aid_set, canonicalize_aid_list

# Global logger
logger = get_logger(__name__)

# --- Global Logger ---
# (Initialized above)

# --- Checkpointing Constants & Functions ---
# Ensure checkpoint directory exists
checkpoint_dir = config.DATA_PROCESSED_DIR
checkpoint_dir.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = checkpoint_dir / "pipeline_checkpoint.json"


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
            pin_memory=config.BI_ENCODER_DATALOADER_PIN_MEMORY,
            prefetch_factor=(
                config.BI_ENCODER_DATALOADER_PREFETCH_FACTOR
                if num_workers > 0
                else None
            ),
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

        # Calculate warmup steps from ratio
        warmup_steps = int(config.BI_ENCODER_WARMUP_RATIO * len(train_examples))

        # Check sentence_transformers version and adjust parameters accordingly
        import sentence_transformers

        st_version = sentence_transformers.__version__
        logger.info(f"[BI-ENCODER] Using sentence_transformers version: {st_version}")

        # Apply runtime compatibility patch for SentenceTransformerTrainer.compute_loss
        try:
            from sentence_transformers.trainer import (
                SentenceTransformerTrainer as _STTrainer,
            )

            _orig_compute_loss = getattr(_STTrainer, "compute_loss", None)

            if callable(_orig_compute_loss):

                def _compat_compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    # Ignore HF's new kwarg on older sbert versions
                    kwargs.pop("num_items_in_batch", None)
                    return _orig_compute_loss(self, model, inputs, return_outputs=return_outputs)

                # Monkey-patch once
                if _STTrainer.compute_loss is not _compat_compute_loss:
                    _STTrainer.compute_loss = _compat_compute_loss
                    logger.info("[BI-ENCODER] Patched SentenceTransformerTrainer.compute_loss for compatibility")
        except Exception as patch_error:
            logger.warning(f"[BI-ENCODER] Could not patch trainer compatibility: {patch_error}")

        # Prepare fit parameters based on version (avoid incompatible kwargs)
        fit_params = {
            "train_objectives": [(train_dataloader, train_loss)],
            "epochs": config.BI_ENCODER_EPOCHS,
            "warmup_steps": warmup_steps,
            "optimizer_params": {
                "lr": config.BI_ENCODER_LR,
                "eps": 1e-6,
            },
            # Remove evaluator for broader compatibility (v4+ may differ)
            # "evaluator": evaluator,
            "output_path": str(config.BI_ENCODER_PATH),
            "show_progress_bar": True,
        }

        # Handle different sentence_transformers versions
        if st_version >= "5.0.0":
            logger.info("[BI-ENCODER] Using sentence_transformers 5.0+ parameters")
            # For version 5.0+, remove evaluator and use newer API
            if "evaluator" in fit_params:
                del fit_params["evaluator"]
            # Add newer parameters for v5.0+
            fit_params["use_amp"] = False  # Disable AMP to avoid issues
            fit_params["checkpoint_save_steps"] = 500
            fit_params["checkpoint_save_total_limit"] = 2
            # Do not pass device here; rely on internal device management
        elif st_version >= "4.0.0":
            logger.info("[BI-ENCODER] Using sentence_transformers 4.0+ parameters")
            # Ensure we don't pass unsupported args like evaluator/device
            if "evaluator" in fit_params:
                del fit_params["evaluator"]
        elif st_version < "2.2.0":
            logger.info("[BI-ENCODER] Using legacy sentence_transformers parameters")
            # Remove any parameters that might cause issues in older versions
            if "evaluator" in fit_params:
                del fit_params["evaluator"]
        else:
            logger.info("[BI-ENCODER] Using standard sentence_transformers parameters")
            # Keep minimal, version-safe args only
            if "evaluator" in fit_params:
                del fit_params["evaluator"]

        model.fit(**fit_params)

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
        from core.utils.aid_utils import canonicalize_aid_ascii

        all_articles = parse_legal_corpus(config.LEGAL_CORPUS_PATH)
        if not all_articles:
            logger.error("[FAISS] No articles found in legal corpus. Aborting.")
            return False

        documents = [article["content"] for article in all_articles]
        # Ensure AIDs are canonicalized before saving mapping for FAISS alignment
        aids = [canonicalize_aid_ascii(article["aid"]) for article in all_articles]

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

    if not dataset_dict["text1"]:
        return None

    # Create dataset
    full_dataset = Dataset.from_dict(dataset_dict)

    # Split into train and validation with robust, version-compatible stratification
    try:
        labels = list(full_dataset["label"]) if "label" in full_dataset.column_names else []
        has_multiple_classes = len(set(labels)) > 1

        if has_multiple_classes:
            # Try encoding label to ClassLabel so stratify_by_column is permitted
            try:
                full_dataset = full_dataset.class_encode_column("label")
            except Exception as e:
                logger.warning(
                    f"[{model_name}] Could not class-encode 'label' column for stratification: {e}"
                )

            # Check feature type dynamically to decide whether we can stratify
            feature_type_name = (
                type(full_dataset.features.get("label")).__name__
                if hasattr(full_dataset, "features") and "label" in full_dataset.features
                else ""
            )

            if feature_type_name == "ClassLabel":
                split_dataset = full_dataset.train_test_split(
                    test_size=config.VALIDATION_SPLIT_RATIO,
                    seed=42,
                    stratify_by_column="label",
                )
            else:
                logger.warning(
                    f"[{model_name}] 'label' is not ClassLabel (got {feature_type_name}); using random split"
                )
                split_dataset = full_dataset.train_test_split(
                    test_size=config.VALIDATION_SPLIT_RATIO, seed=42
                )
        else:
            split_dataset = full_dataset.train_test_split(
                test_size=config.VALIDATION_SPLIT_RATIO, seed=42
            )
    except TypeError:
        # Older datasets versions: no stratification support
        logger.warning(
            f"[{model_name}] 'train_test_split' does not support stratification in this datasets version; using random split"
        )
        split_dataset = full_dataset.train_test_split(
            test_size=config.VALIDATION_SPLIT_RATIO, seed=42
        )

    logger.info(
        f"[{model_name}] Split dataset: {len(split_dataset['train'])} train, {len(split_dataset['test'])} validation"
    )

    return split_dataset


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
                    text1_clean = re.sub(
                        r"[^\w\s\u00C0-\u1EF9]+", " ", str(text1)
                    ).strip()
                    text2_clean = re.sub(
                        r"[^\w\s\u00C0-\u1EF9]+", " ", str(text2)
                    ).strip()

                    # Remove extra whitespace
                    text1_clean = re.sub(r"\s+", " ", text1_clean).strip()
                    text2_clean = re.sub(r"\s+", " ", text2_clean).strip()

                    # Ensure minimum and maximum length
                    if not text1_clean or len(text1_clean) < 3:
                        text1_clean = "text"
                    if not text2_clean or len(text2_clean) < 3:
                        text2_clean = "text"

                    # Strict length limits
                    text1_clean = text1_clean[: max_length // 2]
                    text2_clean = text2_clean[: max_length // 2]

                    cleaned_text1.append(text1_clean)
                    cleaned_text2.append(text2_clean)

                # Tokenize with strict error handling
                result = tokenizer(
                    cleaned_text1,
                    cleaned_text2,
                    truncation="only_second",
                    padding="max_length",
                    max_length=max_length,
                    return_tensors=None,
                )

                # Strict token ID validation
                vocab_size = len(tokenizer)
                unk_id = (
                    tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
                )

                for key in ["input_ids", "attention_mask"]:
                    if key in result:
                        for i, ids in enumerate(result[key]):
                            # Replace ALL invalid tokens with UNK
                            result[key][i] = [
                                (
                                    unk_id
                                    if token_id >= vocab_size or token_id < 0
                                    else token_id
                                )
                                for token_id in ids
                            ]

                return result
            except Exception as e:
                logger.error(f"[{model_log_name}] Critical preprocessing error: {e}")
                # Return safe fallback
                return {
                    "input_ids": [
                        [tokenizer.cls_token_id]
                        + [unk_id] * (max_length - 2)
                        + [tokenizer.sep_token_id]
                    ],
                    "attention_mask": [[1] * max_length],
                    "labels": [0],
                }

        # Handle dataset splits - training_data is already split from _prepare_reranker_data
        if (
            isinstance(training_data, dict)
            and "train" in training_data
            and "test" in training_data
        ):
            # Dataset is already split
            train_dataset = training_data["train"].map(
                preprocess_function, batched=True
            )
            eval_dataset = training_data["test"].map(preprocess_function, batched=True)
        else:
            # Fallback: split the dataset here
            dataset_splits = training_data.train_test_split(test_size=0.1, seed=42)
            train_dataset = dataset_splits["train"].map(
                preprocess_function, batched=True
            )
            eval_dataset = dataset_splits["test"].map(preprocess_function, batched=True)

        # Add early stopping callback based on model type (compat with older transformers)
        callbacks = []
        try:
            # Only add when we know Trainer can evaluate; otherwise it asserts
            if hasattr(training_args, "evaluation_strategy") and str(getattr(training_args, "evaluation_strategy", "")):
                if "Light-Reranker" in model_log_name:
                    callbacks.append(
                        EarlyStoppingCallback(
                            early_stopping_patience=config.LIGHT_RERANKER_EARLY_STOPPING_PATIENCE,
                            early_stopping_threshold=config.LIGHT_RERANKER_EARLY_STOPPING_THRESHOLD,
                        )
                    )
                elif "Cross-Encoder" in model_log_name:
                    callbacks.append(
                        EarlyStoppingCallback(
                            early_stopping_patience=config.CROSS_ENCODER_EARLY_STOPPING_PATIENCE,
                            early_stopping_threshold=config.CROSS_ENCODER_EARLY_STOPPING_THRESHOLD,
                        )
                    )
        except Exception as e:
            logger.warning(
                f"[{model_log_name}] Could not add EarlyStoppingCallback: {e}"
            )
            # Continue without early stopping if there's an issue
            callbacks = []

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=lambda p: {
                "accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()
            },
            callbacks=callbacks,
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
                logger.error(
                    f"[{model_log_name}] CUDA/Memory error during training: {e}"
                )

                # Try with reduced batch size first
                logger.info(f"[{model_log_name}] Attempting with reduced batch size...")
                training_args.per_device_train_batch_size = max(
                    1, training_args.per_device_train_batch_size // 2
                )
                training_args.per_device_eval_batch_size = max(
                    1, training_args.per_device_eval_batch_size // 2
                )

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
                    logger.error(
                        f"[{model_log_name}] Still failing with reduced batch size: {e2}"
                    )
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
# --- Compatibility helper for TrainingArguments across transformers versions ---
def build_training_args_compat(
    *,
    output_dir: str,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    learning_rate: float,
    warmup_steps: int,
    eval_steps: int,
    save_steps: int,
    fp16: bool,
    gradient_accumulation_steps: int,
    dataloader_num_workers: int = None,
    dataloader_pin_memory: bool = None,
    dataloader_prefetch_factor: int = None,
    metric_for_best_model: str = None,
    greater_is_better: bool = None,
    load_best_model_at_end: bool = None,
):
    """Build TrainingArguments with graceful degradation for older transformers."""
    from transformers import TrainingArguments

    init_params = inspect.signature(TrainingArguments.__init__).parameters
    accepted = set(init_params.keys())

    # Base kwargs
    kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "logging_steps": max(1, eval_steps // 2) if isinstance(eval_steps, int) and eval_steps > 0 else 10,
        "save_steps": save_steps,
        "fp16": fp16,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }

    # Optional params included only if supported
    optional_params = {
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_pin_memory": dataloader_pin_memory,
        # Newer APIs only; include if available
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": greater_is_better,
        "load_best_model_at_end": load_best_model_at_end,
        # Many older versions don't support these; we skip by default
        # "evaluation_strategy": "steps",
        # "save_strategy": "steps",
        # "remove_unused_columns": False,
        # "report_to": "none",
    }

    for key, value in optional_params.items():
        if value is not None and key in accepted:
            kwargs[key] = value

    # Strategies: ensure consistency if possible
    can_set_eval = "evaluation_strategy" in accepted
    can_set_save = "save_strategy" in accepted
    if can_set_eval and can_set_save:
        # align both to steps
        kwargs["evaluation_strategy"] = "steps"
        kwargs["save_strategy"] = "steps"
    else:
        # Cannot set strategies consistently; disable best model at end to avoid ValueError
        if "load_best_model_at_end" in kwargs:
            kwargs["load_best_model_at_end"] = False
        # Fallback for very old versions: use evaluate_during_training if exists
        if "evaluation_strategy" not in accepted and "evaluate_during_training" in accepted and eval_steps:
            kwargs["evaluate_during_training"] = True

    # Finally, filter strictly to accepted keys
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}

    return TrainingArguments(**filtered_kwargs)


def run_comprehensive_evaluation():
    """Chạy đánh giá toàn diện ngay sau huấn luyện với đầy đủ Tiers (1/2/3 và Cascaded)."""
    logger.info("[EVAL] Starting comprehensive evaluation...")
    try:
        # 1) Khởi tạo pipeline (bật cascaded để sẵn sàng Light + Strong)
        logger.info("[EVAL] Initializing pipeline...")
        pipeline = LegalQAPipeline(use_ensemble=True, use_cascaded_reranking=True)
        if not pipeline.is_ready:
            logger.error("[EVAL] Pipeline is not ready. Cannot run evaluation.")
            return False

        # 2) Tải validation split và canonicalize ground-truth AIDs
        logger.info("[EVAL] Loading validation data...")
        if not config.VAL_SPLIT_JSON_PATH.exists():
            logger.error(f"[EVAL] Validation data not found at {config.VAL_SPLIT_JSON_PATH}")
            return False
        with open(config.VAL_SPLIT_JSON_PATH, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        queries = [item.get("question", "") for item in val_data]
        ground_truth_sets = [canonicalize_aid_set(item.get("relevant_aids", [])) for item in val_data]
        logger.info(f"[EVAL] Loaded {len(queries)} validation queries")

        # Coverage check: đảm bảo GT có trong index
        try:
            if config.INDEX_TO_AID_PATH.exists():
                with open(config.INDEX_TO_AID_PATH, "r", encoding="utf-8") as f:
                    index_aids = set(canonicalize_aid_list(json.load(f)))
                total_gt = sum(len(s) for s in ground_truth_sets)
                present = sum(1 for s in ground_truth_sets for aid in s if aid in index_aids)
                coverage = (present / total_gt * 100) if total_gt else 0.0
                logger.info(f"[EVAL] Ground-truth coverage in index: {present}/{total_gt} ({coverage:.2f}%)")
                if coverage < 50.0:
                    logger.warning("[EVAL] Low index coverage. Consider rebuilding FAISS index.")
            else:
                logger.warning(f"[EVAL] INDEX_TO_AID file not found at {config.INDEX_TO_AID_PATH}")
        except Exception as e:
            logger.warning(f"[EVAL] Coverage check failed: {e}")

        # 3) Tier 1: Retrieval (batch) cho tốc độ
        logger.info("[EVAL] Tier 1 - Batch retrieval...")
        retrieved_aids_batch, distances_batch = pipeline.retrieve_batch(queries, config.TOP_K_RETRIEVAL)
        retrieval_scores = []
        for distances in distances_batch:
            if distances is None or len(distances) == 0:
                retrieval_scores.append([])
                continue
            max_dist = max(distances) if len(distances) > 0 else 1.0
            retrieval_scores.append([1.0 - (d / max_dist) if max_dist > 0 else 0.0 for d in distances])
        evaluator = BatchEvaluator(k_values=[1, 3, 5, 10, 20, 50])
        tier1_metrics = evaluator.evaluate_batch(queries, ground_truth_sets, retrieved_aids_batch) or {}

        # 4) Tier 2: Light Reranker (light-only, không strong)
        logger.info("[EVAL] Tier 2 - Light Reranker (only)...")
        light_only_aids_batch = []
        light_only_dists_batch = []
        for q, aids, dists in zip(queries, retrieved_aids_batch, distances_batch):
            try:
                light_aids, light_dists = pipeline.rerank_light(q, aids, dists, top_k_light=config.TOP_K_LIGHT_RERANKING)
                light_only_aids_batch.append(light_aids)
                light_only_dists_batch.append(light_dists)
            except Exception as e:
                logger.warning(f"[EVAL] Light reranker failed: {e}")
                light_only_aids_batch.append([])
                light_only_dists_batch.append([])
        light_metrics = evaluator.evaluate_batch(queries, ground_truth_sets, light_only_aids_batch) or {}

        # 5) Tier 3: Strong-only Reranker (không dùng light)
        logger.info("[EVAL] Tier 3 - Strong-only reranker...")
        strong_only_predictions = []
        strong_only_scores = []
        for q, aids, dists in zip(queries, retrieved_aids_batch, distances_batch):
            try:
                results = pipeline.rerank(q, aids, dists)
                strong_only_predictions.append(results)
                strong_only_scores.append([r.get("rerank_score", 0.0) for r in results])
            except Exception as e:
                logger.warning(f"[EVAL] Strong rerank failed: {e}")
                strong_only_predictions.append([])
                strong_only_scores.append([])
        strong_only_aids_batch = [[r.get("aid") for r in preds] for preds in strong_only_predictions]
        tier3_metrics = evaluator.evaluate_batch(queries, ground_truth_sets, strong_only_aids_batch) or {}

        # 6) Cascaded: Light + Strong (rerank trên top từ light)
        logger.info("[EVAL] Cascaded - Light + Strong reranker...")
        cascaded_predictions = []
        cascaded_scores = []
        for q, light_aids, light_dists in zip(queries, light_only_aids_batch, light_only_dists_batch):
            try:
                results = pipeline.rerank(q, light_aids, light_dists)
                cascaded_predictions.append(results)
                cascaded_scores.append([r.get("rerank_score", 0.0) for r in results])
            except Exception as e:
                logger.warning(f"[EVAL] Cascaded rerank failed: {e}")
                cascaded_predictions.append([])
                cascaded_scores.append([])
        cascaded_aids_batch = [[r.get("aid") for r in preds] for preds in cascaded_predictions]
        cascaded_metrics = evaluator.evaluate_batch(queries, ground_truth_sets, cascaded_aids_batch) or {}

        # 7) Per-query details
        logger.info("[EVAL] Building per-query detailed results...")
        per_query_results = []
        for i, (q, gt_set) in enumerate(zip(queries, ground_truth_sets)):
            # Retrieval
            ret_aids = canonicalize_aid_list(retrieved_aids_batch[i]) if i < len(retrieved_aids_batch) else []
            ret_scores = retrieval_scores[i] if i < len(retrieval_scores) else []
            ret_precision = (len(set(ret_aids) & gt_set) / len(ret_aids)) if ret_aids else 0.0
            ret_recall = (len(set(ret_aids) & gt_set) / len(gt_set)) if gt_set else 0.0
            ret_f1 = (2 * ret_precision * ret_recall / (ret_precision + ret_recall)) if (ret_precision + ret_recall) > 0 else 0.0

            # Light-only
            light_aids = canonicalize_aid_list(light_only_aids_batch[i]) if i < len(light_only_aids_batch) else []
            light_precision = (len(set(light_aids) & gt_set) / len(light_aids)) if light_aids else 0.0
            light_recall = (len(set(light_aids) & gt_set) / len(gt_set)) if gt_set else 0.0
            light_f1 = (2 * light_precision * light_recall / (light_precision + light_recall)) if (light_precision + light_recall) > 0 else 0.0

            # Strong-only
            strong_results = strong_only_predictions[i] if i < len(strong_only_predictions) else []
            strong_aids = canonicalize_aid_list([r.get("aid") for r in strong_results]) if strong_results else []
            strong_scores = strong_only_scores[i] if i < len(strong_only_scores) else []
            strong_precision = (len(set(strong_aids) & gt_set) / len(strong_aids)) if strong_aids else 0.0
            strong_recall = (len(set(strong_aids) & gt_set) / len(gt_set)) if gt_set else 0.0
            strong_f1 = (2 * strong_precision * strong_recall / (strong_precision + strong_recall)) if (strong_precision + strong_recall) > 0 else 0.0

            # Cascaded
            casc_results = cascaded_predictions[i] if i < len(cascaded_predictions) else []
            casc_aids = canonicalize_aid_list([r.get("aid") for r in casc_results]) if casc_results else []
            casc_scores = cascaded_scores[i] if i < len(cascaded_scores) else []
            casc_precision = (len(set(casc_aids) & gt_set) / len(casc_aids)) if casc_aids else 0.0
            casc_recall = (len(set(casc_aids) & gt_set) / len(gt_set)) if gt_set else 0.0
            casc_f1 = (2 * casc_precision * casc_recall / (casc_precision + casc_recall)) if (casc_precision + casc_recall) > 0 else 0.0

            per_query_results.append({
                "query_id": i,
                "query": q,
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
                "tier2_light_results": {
                    "aids": light_aids,
                    "scores": [],
                    "precision": light_precision,
                    "recall": light_recall,
                    "f1": light_f1,
                    "found_relevant": len(set(light_aids) & gt_set),
                },
                "tier3_strong_only_results": {
                    "aids": strong_aids,
                    "scores": strong_scores,
                    "precision": strong_precision,
                    "recall": strong_recall,
                    "f1": strong_f1,
                    "found_relevant": len(set(strong_aids) & gt_set),
                },
                "cascaded_results": {
                    "aids": casc_aids,
                    "scores": casc_scores,
                    "precision": casc_precision,
                    "recall": casc_recall,
                    "f1": casc_f1,
                    "found_relevant": len(set(casc_aids) & gt_set),
                },
                "improvement": {
                    "tier2_over_tier1": {"precision": light_precision - ret_precision, "recall": light_recall - ret_recall, "f1": light_f1 - ret_f1},
                    "tier3_over_tier1": {"precision": strong_precision - ret_precision, "recall": strong_recall - ret_recall, "f1": strong_f1 - ret_f1},
                    "cascaded_over_tier2": {"precision": casc_precision - light_precision, "recall": casc_recall - light_recall, "f1": casc_f1 - light_f1},
                    "cascaded_over_tier3": {"precision": casc_precision - strong_precision, "recall": casc_recall - strong_recall, "f1": casc_f1 - strong_f1},
                    "cascaded_over_tier1": {"precision": casc_precision - ret_precision, "recall": casc_recall - ret_recall, "f1": casc_f1 - ret_f1},
                },
            })

        # 8) Metadata & report
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "training_post_eval",
            "total_queries": len(queries),
            "aid_normalization": "canonical_ascii",
            "data_source": str(config.VAL_SPLIT_JSON_PATH),
            "pipeline_config": {
                "top_k_retrieval": config.TOP_K_RETRIEVAL,
                "top_k_light": getattr(config, "TOP_K_LIGHT_RERANKING", None),
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
        }

        reporter = EvaluationReporter()
        report = reporter.create_comprehensive_report(
            retrieval_metrics=tier1_metrics,
            reranking_metrics=tier3_metrics,
            per_query_results=per_query_results,
            metadata=metadata,
            cascaded_metrics=cascaded_metrics,
            light_metrics=light_metrics,
        )
        reporter.display_summary(report)
        report_path = reporter.save_report(report)

        logger.info(f"📄 Detailed report saved to: {report_path}")
        logger.info("=" * 80)
        return True

    except Exception as e:
        logger.error(f"[EVAL] Comprehensive evaluation failed: {e}", exc_info=True)
        return False


# --- Main Pipeline Execution ---


def main():
    """Hàm chính điều khiển toàn bộ pipeline huấn luyện và đánh giá."""
    # --- Config Reloading for Performance Mode ---
    # Reload config to ensure latest performance mode settings are applied
    try:
        importlib.reload(config)
        logger.info(f"[CONFIG] Reloaded config with PERFORMANCE_MODE: {config.PERFORMANCE_MODE}")
        logger.info(f"[CONFIG] Bi-Encoder epochs: {config.BI_ENCODER_EPOCHS}")
        logger.info(f"[CONFIG] Cross-Encoder epochs: {config.CROSS_ENCODER_EPOCHS}")
        logger.info(f"[CONFIG] Light Reranker epochs: {config.LIGHT_RERANKER_EPOCHS}")
        
        # Double-check environment variable (ensure scripts can be called directly with --mode)
        env_performance_mode = os.getenv("LAWBOT_PERFORMANCE_MODE", "quality")
        if env_performance_mode != config.PERFORMANCE_MODE:
            logger.warning(
                f"[CONFIG] Environment variable LAWBOT_PERFORMANCE_MODE={env_performance_mode} but config.PERFORMANCE_MODE={config.PERFORMANCE_MODE}"
            )
            logger.info("[CONFIG] Forcing config reload with environment variable...")
            # Force reload by clearing module cache
            for m in ('config', 'config_fast', 'config_quality', 'config_base'):
                if m in sys.modules:
                    del sys.modules[m]
            # Re-import config without creating a local binding that shadows the global
            reloaded_config = importlib.import_module('config')
            globals()['config'] = reloaded_config
            logger.info(f"[CONFIG] After force reload - PERFORMANCE_MODE: {config.PERFORMANCE_MODE}")
            
    except Exception as e:
        logger.warning(f"[CONFIG] Could not reload config: {e}")
        # Ensure config is available even if reload fails
        try:
            reloaded_config = importlib.import_module('config')
            globals()['config'] = reloaded_config
        except Exception:
            logger.error("Could not import config module")
            return False
    
    logger.info("=" * 80)
    logger.info("STARTING: Model Training & Evaluation Pipeline v8.0")
    logger.info(f"PERFORMANCE MODE: {config.PERFORMANCE_MODE.upper()}")
    logger.info("=" * 80)
    logger.info("Pipeline Overview:")
    logger.info("   - Step 1: Bi-Encoder Training (Sentence Transformers)")
    logger.info("   - Step 2: FAISS Index Building (Vector Search)")
    logger.info("   - Step 3: Cross-Encoder Training (Sequence Classification)")
    logger.info("   - Step 4: Light Reranker Training (Fast Reranking)")
    logger.info("   - Step 5: Comprehensive Evaluation (Full Metrics)")
    logger.info("=" * 80)

    # Ensure all necessary directories exist
    logger.info("Creating necessary directories...")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    config.BI_ENCODER_PATH.mkdir(parents=True, exist_ok=True)
    config.CROSS_ENCODER_PATH.mkdir(parents=True, exist_ok=True)
    config.LIGHT_RERANKER_PATH.mkdir(parents=True, exist_ok=True)
    logger.info("Directories created successfully")

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
                args = build_training_args_compat(
                    output_dir=str(config.CROSS_ENCODER_PATH),
                    num_train_epochs=config.CROSS_ENCODER_EPOCHS,
                    per_device_train_batch_size=config.CROSS_ENCODER_BATCH_SIZE,
                    learning_rate=config.CROSS_ENCODER_LR,
                    warmup_steps=config.CROSS_ENCODER_WARMUP_RATIO,
                    eval_steps=config.CROSS_ENCODER_EVAL_STEPS,
                    save_steps=config.CROSS_ENCODER_EVAL_STEPS * 2,
                    fp16=config.FP16_TRAINING,
                    gradient_accumulation_steps=config.CROSS_ENCODER_GRADIENT_ACCUMULATION_STEPS,
                    dataloader_num_workers=config.CROSS_ENCODER_DATALOADER_NUM_WORKERS,
                    dataloader_pin_memory=False,
                    dataloader_prefetch_factor=config.CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR,
                    metric_for_best_model=None,
                    greater_is_better=None,
                    load_best_model_at_end=False,
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
            logger.info(
                "STEP 3: Cross-Encoder Training... [SKIPPED - Already complete]"
            )

        # --- Step 4: Light Reranker Training ---
        if not is_step_complete(checkpoint_state, "train_light_reranker"):
            logger.info("STEP 4: Light Reranker Training...")
            dataset = _prepare_reranker_data(reranker_data, "Light-Reranker")
            if dataset:
                args = build_training_args_compat(
                    output_dir=str(config.LIGHT_RERANKER_PATH),
                    num_train_epochs=config.LIGHT_RERANKER_EPOCHS,
                    per_device_train_batch_size=config.LIGHT_RERANKER_BATCH_SIZE,
                    learning_rate=config.LIGHT_RERANKER_LR,
                    warmup_steps=int(
                        config.LIGHT_RERANKER_WARMUP_RATIO * len(dataset["train"])
                    ),
                    eval_steps=config.LIGHT_RERANKER_EVAL_STEPS,
                    save_steps=config.LIGHT_RERANKER_EVAL_STEPS * 2,
                    fp16=config.FP16_TRAINING,
                    gradient_accumulation_steps=config.LIGHT_RERANKER_GRADIENT_ACCUMULATION_STEPS,
                    dataloader_num_workers=config.LIGHT_RERANKER_DATALOADER_NUM_WORKERS,
                    dataloader_pin_memory=config.CROSS_ENCODER_DATALOADER_PIN_MEMORY,
                    dataloader_prefetch_factor=config.CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR,
                    metric_for_best_model=None,
                    greater_is_better=None,
                    load_best_model_at_end=False,
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
            logger.info(
                "STEP 4: Light Reranker Training... [SKIPPED - Already complete]"
            )

        # --- Step 5: Comprehensive Evaluation ---
        if not is_step_complete(checkpoint_state, "run_evaluation"):
            logger.info("STEP 5: Comprehensive Evaluation...")
            if not run_comprehensive_evaluation():
                logger.warning(
                    "Evaluation run failed, but training steps are complete."
                )
            mark_step_complete(checkpoint_state, "run_evaluation")
        else:
            logger.info(
                "STEP 5: Comprehensive Evaluation... [SKIPPED - Already complete]"
            )

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
