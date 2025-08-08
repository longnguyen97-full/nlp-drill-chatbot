#!/usr/bin/env python3
"""
Training Data Preparation Pipeline - Advanced Optimization v7.1 (Fixed)
=======================================================================

Script nay chuan bi training data cho Bi-Encoder va Cross-Encoder
voi cac ky thuat toi uu nang cao:
- Advanced Hard Negative Mining with robust model loading
- AI-Powered Augmentation (Back-Translation & Query Generation)

Tac gia: LawBot Team
Phien ban: Advanced Optimization v7.1 (Fixed)
"""

import json
import logging
import random
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import faiss

# Them thu muc goc vao path
import sys
import os

from transformers import AutoModel, AutoTokenizer
from sentence_transformers.models import Pooling, WordEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from core.logging_system import get_logger

# Sử dụng logger đã được setup từ pipeline chính
logger = get_logger(__name__)


def load_processed_data():
    """Load du lieu da duoc xu ly tu buoc truoc"""
    logger.info("[LOAD] Loading processed data from previous step...")

    # Load train split
    with open(config.TRAIN_SPLIT_JSON_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # Load aid map
    with open(config.AID_MAP_PATH, "rb") as f:
        aid_map = pickle.load(f)

    logger.info(f"[LOAD] Loaded {len(train_data)} training samples")
    logger.info(f"[LOAD] Loaded {len(aid_map)} AIDs")

    return train_data, aid_map


def create_initial_triplets(train_data: List[Dict], aid_map: Dict) -> List[Dict]:
    """Tao triplets ban dau cho Bi-Encoder (easy negatives)."""
    logger.info("[TRIPLET] Creating initial triplets for Bi-Encoder...")
    triplets = []
    all_aids = list(aid_map.keys())

    for sample in train_data:
        question = sample.get("question")
        relevant_aids = sample.get("relevant_aids", [])

        if not question or not relevant_aids:
            continue

        positive_contents = [
            aid_map.get(aid) for aid in relevant_aids if aid in aid_map
        ]
        negative_aids_pool = [aid for aid in all_aids if aid not in relevant_aids]

        if not positive_contents or not negative_aids_pool:
            continue

        for positive_content in positive_contents:
            negative_aid = random.choice(negative_aids_pool)
            negative_content = aid_map[negative_aid]
            triplets.append(
                {
                    "anchor": question,
                    "positive": positive_content,
                    "negative": negative_content,
                }
            )

    logger.info(f"[TRIPLET] Created {len(triplets)} initial triplets.")
    return triplets


def load_optimized_model_for_hard_negative_mining():
    """
    Load model tối ưu để tìm hard negatives.
    Ưu tiên model đã được fine-tune, sau đó đến model DAPT, cuối cùng là model gốc.
    Thực hiện sửa lỗi vocab_size một cách tường minh.
    """
    logger.info("[HARD_NEG] Loading optimized model for hard negative mining...")

    # Lựa chọn 1: Thử load Bi-Encoder đã được huấn luyện (tốt nhất)
    bi_encoder_path = config.BI_ENCODER_PATH
    if bi_encoder_path.exists() and any(bi_encoder_path.iterdir()):
        try:
            logger.info(
                f"[HARD_NEG] Found existing Bi-Encoder model at '{bi_encoder_path}', loading..."
            )
            model = SentenceTransformer(str(bi_encoder_path))
            logger.info("[HARD_NEG] Successfully loaded existing Bi-Encoder model.")
            return model, "existing_bi_encoder"
        except Exception as e:
            logger.warning(
                f"[HARD_NEG] Failed to load existing Bi-Encoder: {e}. Trying next option."
            )

    # Lựa chọn 2: Thử load PhoBERT-Law từ DAPT (tốt) và sửa lỗi vocab size
    phobert_law_path = str(config.PHOBERT_LAW_PATH)
    if Path(phobert_law_path).exists() and any(Path(phobert_law_path).iterdir()):
        try:
            logger.info(
                f"[HARD_NEG] Found PhoBERT-Law model at '{phobert_law_path}', loading manually..."
            )

            # Nạp thủ công để kiểm soát
            tokenizer = AutoTokenizer.from_pretrained(phobert_law_path)
            auto_model = AutoModel.from_pretrained(phobert_law_path)

            # **SỬA LỖI QUAN TRỌNG TẠI ĐÂY**
            if len(tokenizer) != auto_model.config.vocab_size:
                logger.warning(
                    f"[HARD_NEG] Vocab size mismatch detected. Resizing model embeddings to {len(tokenizer)}."
                )
                auto_model.resize_token_embeddings(len(tokenizer))

            # Tạo SentenceTransformer từ các thành phần đã được sửa lỗi
            word_embedding_model = WordEmbeddings(
                model_name_or_path=None, model=auto_model, tokenizer=tokenizer
            )
            pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

            logger.info(
                "[HARD_NEG] Successfully loaded and configured PhoBERT-Law model."
            )
            return model, "phobert_law_fixed"
        except Exception as e:
            logger.warning(
                f"[HARD_NEG] Failed to load PhoBERT-Law: {e}. Trying next option."
            )

    # Lựa chọn 3: Fallback to base model (ít tối ưu nhất)
    try:
        logger.info(
            f"[HARD_NEG] No custom models found. Using base model '{config.BI_ENCODER_MODEL_NAME}'..."
        )
        model = SentenceTransformer(config.BI_ENCODER_MODEL_NAME)
        logger.info("[HARD_NEG] Successfully loaded base model.")
        return model, "base_model"
    except Exception as e:
        logger.error(
            f"[HARD_NEG] CRITICAL: Failed to load ANY model. Error: {e}", exc_info=True
        )
        return None, "none"


def find_hard_negatives(
    model: SentenceTransformer, train_data: List[Dict], aid_map: Dict, model_type: str
) -> List[Dict]:
    """Tìm các ví dụ "âm tính khó" (hard negatives) bằng model đã được tối ưu cao cấp."""
    logger.info(
        f"[HARD_NEG] Finding advanced hard negatives using '{model_type}' model..."
    )

    if model is None:
        logger.error("[HARD_NEG] Model is None. Skipping hard negative mining.")
        return []

    hard_negatives_triplets = []
    all_contents = list(aid_map.values())
    all_aids = list(aid_map.keys())

    logger.info(
        f"[HARD_NEG] Creating optimized embeddings for all {len(all_contents)} legal articles..."
    )

    # OPTIMIZATION: Adaptive batch sizing based on memory
    import psutil

    memory = psutil.virtual_memory()
    available_memory_gb = memory.available / (1024**3)

    # Adaptive batch size based on available memory
    if available_memory_gb >= 8:
        batch_size = 128
    elif available_memory_gb >= 4:
        batch_size = 64
    else:
        batch_size = 32

    logger.info(
        f"[HARD_NEG] Available memory: {available_memory_gb:.1f} GB, using batch size: {batch_size}"
    )

    # Add progress tracking
    total_batches = (len(all_contents) + batch_size - 1) // batch_size
    logger.info(f"[HARD_NEG] Processing {total_batches} batches...")

    try:
        embeddings = model.encode(
            all_contents,
            show_progress_bar=True,
            batch_size=batch_size,
            convert_to_numpy=True,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning("[HARD_NEG] Out of memory, reducing batch size...")
            batch_size = max(8, batch_size // 2)
            embeddings = model.encode(
                all_contents,
                show_progress_bar=True,
                batch_size=batch_size,
                convert_to_numpy=True,
            )
        else:
            raise e

    dimension = embeddings.shape[1]
    faiss.normalize_L2(embeddings)

    # OPTIMIZATION: Use more efficient FAISS index
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))
    logger.info(f"[HARD_NEG] FAISS index created with {index.ntotal} vectors.")

    questions = [
        sample["question"] for sample in train_data if sample.get("relevant_aids")
    ]
    logger.info(
        f"[HARD_NEG] Processing {len(questions)} questions for hard negative mining..."
    )

    try:
        question_embeddings = model.encode(
            questions, show_progress_bar=True, batch_size=batch_size
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(
                "[HARD_NEG] Out of memory during question encoding, reducing batch size..."
            )
            batch_size = max(4, batch_size // 2)
            question_embeddings = model.encode(
                questions, show_progress_bar=True, batch_size=batch_size
            )
        else:
            raise e

    faiss.normalize_L2(question_embeddings)

    # OPTIMIZATION: Increase top-k for better negative mining
    top_k = min(50, len(all_contents))  # Increased from config value
    scores_batch, indices_batch = index.search(
        question_embeddings.astype(np.float32), top_k
    )

    sample_idx = 0
    processed_samples = 0

    for i, sample in enumerate(train_data):
        if not sample.get("relevant_aids"):
            continue

        question = sample["question"]
        relevant_aids = set(sample["relevant_aids"])

        scores, indices = scores_batch[sample_idx], indices_batch[sample_idx]
        sample_idx += 1

        # OPTIMIZATION: Better negative selection strategy
        hard_neg_candidates = []
        start, end = 5, 25  # Focus on high-similarity but incorrect candidates

        for j in range(start, min(end, len(indices))):
            candidate_aid = all_aids[indices[j]]
            if candidate_aid not in relevant_aids:
                hard_neg_candidates.append(candidate_aid)

        # OPTIMIZATION: Add diversity to negative samples
        selected_negatives = set()
        for positive_aid in relevant_aids:
            if positive_aid not in aid_map:
                continue
            positive_content = aid_map[positive_aid]

            # Select diverse negatives
            neg_count = 0
            for neg_aid in hard_neg_candidates:
                if neg_count >= 3:  # Limit per positive for balance
                    break

                if neg_aid not in selected_negatives:
                    hard_negatives_triplets.append(
                        {
                            "anchor": question,
                            "positive": positive_content,
                            "negative": aid_map[neg_aid],
                            "is_hard_negative": True,
                        }
                    )
                    selected_negatives.add(neg_aid)
                    neg_count += 1

        processed_samples += 1
        if processed_samples % 100 == 0:
            logger.info(
                f"[HARD_NEG] Processed {processed_samples}/{len(train_data)} samples..."
            )

    # Memory cleanup
    del embeddings, question_embeddings, index
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        f"[HARD_NEG] Found {len(hard_negatives_triplets)} high-quality hard negative triplets."
    )
    return hard_negatives_triplets


def create_final_dataset(initial_triplets, hard_negatives, train_data, aid_map):
    """Tạo bộ dữ liệu cuối cùng cho Bi-Encoder và Cross-Encoder với quality filtering."""
    logger.info("[DATASET] Creating final datasets with quality filtering...")

    # --- Bi-Encoder Dataset ---
    bi_encoder_data = initial_triplets + hard_negatives
    random.shuffle(bi_encoder_data)

    # --- Cross-Encoder Dataset với quality filtering ---
    cross_encoder_data = []
    hard_neg_map = {}
    for triplet in hard_negatives:
        q = triplet["anchor"]
        if q not in hard_neg_map:
            hard_neg_map[q] = set()
        hard_neg_map[q].add(triplet["negative"])

    all_aids = set(aid_map.keys())
    quality_filtered_count = 0

    for sample in train_data:
        question = sample["question"]
        relevant_aids = set(sample.get("relevant_aids", []))
        if not relevant_aids:
            continue

        # Quality filtering for question
        if len(question.strip()) < 10 or len(question.strip()) > 500:
            quality_filtered_count += 1
            continue

        # Positive pairs với quality check
        for aid in relevant_aids:
            if aid in aid_map:
                content = aid_map[aid]
                # Quality filtering for content
                if len(content.strip()) >= 20 and len(content.strip()) <= 3000:
                    cross_encoder_data.append(
                        {"texts": [question, content], "label": 1}
                    )

        # Hard Negative pairs với quality check
        hard_negs_for_q = {
            aid_map.get(aid) for aid in hard_neg_map.get(question, set())
        }
        for neg_content in hard_negs_for_q:
            if (
                neg_content
                and len(neg_content.strip()) >= 20
                and len(neg_content.strip()) <= 3000
            ):
                cross_encoder_data.append(
                    {"texts": [question, neg_content], "label": 0}
                )

        # Random Negative pairs với quality check
        num_to_add = max(1, len(relevant_aids) // 2)  # Limit negative samples
        negative_pool = [aid for aid in all_aids - relevant_aids if aid in aid_map]
        if negative_pool and num_to_add > 0:
            random_negs = random.sample(
                negative_pool, min(num_to_add, len(negative_pool))
            )
            for neg_aid in random_negs:
                content = aid_map[neg_aid]
                if len(content.strip()) >= 20 and len(content.strip()) <= 3000:
                    cross_encoder_data.append(
                        {"texts": [question, content], "label": 0}
                    )

    random.shuffle(cross_encoder_data)

    logger.info(f"Final Bi-Encoder samples: {len(bi_encoder_data)}")
    logger.info(f"Final Cross-Encoder samples: {len(cross_encoder_data)}")
    logger.info(f"Quality filtered samples: {quality_filtered_count}")

    return bi_encoder_data, cross_encoder_data


def save_training_data(bi_encoder_data, cross_encoder_data):
    """Lưu dữ liệu huấn luyện đã được xử lý ra file."""
    logger.info("[SAVE] Saving final training data...")
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Sử dụng tên file theo config để đảm bảo tính nhất quán
    bi_encoder_path = config.BI_ENCODER_TRAIN_MIXED_PATH
    cross_encoder_path = config.TRAIN_PAIRS_MIXED_PATH

    with open(bi_encoder_path, "w", encoding="utf-8") as f:
        for item in bi_encoder_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(cross_encoder_path, "w", encoding="utf-8") as f:
        for item in cross_encoder_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"[SAVE] Bi-Encoder data saved to: {bi_encoder_path}")
    logger.info(f"[SAVE] Cross-Encoder data saved to: {cross_encoder_path}")


def run_prepare_data_pipeline():
    """Chạy toàn bộ pipeline chuẩn bị dữ liệu."""
    logger.info("=" * 80)
    logger.info("STARTING: Training Data Preparation Pipeline v7.1")
    logger.info("=" * 80)
    logger.info("Pipeline Overview:")
    logger.info("   - Step 1: Load processed data from previous step")
    logger.info("   - Step 2: Create initial triplets (easy negatives)")
    logger.info("   - Step 3: Load model for hard negative mining")
    logger.info("   - Step 4: Mine hard negatives using semantic similarity")
    logger.info("   - Step 5: Create final training datasets")
    logger.info("   - Step 6: Save training data to files")
    logger.info("=" * 80)

    try:
        # Step 1: Load data
        train_data, aid_map = load_processed_data()

        # Step 2: Create initial "easy negative" triplets
        initial_triplets = create_initial_triplets(train_data, aid_map)

        # Step 3: Load or build a model for mining hard negatives
        model, model_type = load_optimized_model_for_hard_negative_mining()
        if not model:
            raise RuntimeError("Could not load any model for hard negative mining.")

        # Step 4: Mine hard negatives
        hard_negatives = find_hard_negatives(model, train_data, aid_map, model_type)

        # Clean up model to free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 5: Create final training sets
        bi_encoder_data, cross_encoder_data = create_final_dataset(
            initial_triplets, hard_negatives, train_data, aid_map
        )

        # Step 6: Save data to files
        save_training_data(bi_encoder_data, cross_encoder_data)

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"- Model used for mining: {model_type}")
        logger.info(f"- Total Bi-Encoder samples: {len(bi_encoder_data)}")
        logger.info(f"- Total Cross-Encoder samples: {len(cross_encoder_data)}")
        logger.info("=" * 80)
        return True

    except Exception as e:
        logger.error(f"PIPELINE FAILED: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    if run_prepare_data_pipeline():
        logger.info("Training data is ready for the next step.")
        sys.exit(0)
    else:
        logger.error("Pipeline failed. Please check the logs.")
        sys.exit(1)
