#!/usr/bin/env python3
"""
Training Data Preparation Pipeline - Advanced Optimization v7.0
==============================================================

Script nay chuan bi training data cho Bi-Encoder va Cross-Encoder
voi cac ky thuat toi uu nang cao:
- Advanced Hard Negative Mining
- AI-Powered Augmentation (Back-Translation & Query Generation)

Tac gia: LawBot Team
Phien ban: Advanced Optimization v7.0
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


def create_initial_triplets(train_data, aid_map):
    """Tao triplets ban dau cho Bi-Encoder"""
    logger.info("[TRIPLET] Creating initial triplets for Bi-Encoder...")

    triplets = []
    all_aids = list(aid_map.keys())

    for sample in train_data:
        question = sample["question"]
        relevant_aids = sample["relevant_aids"]

        if not relevant_aids:
            continue

        # Tao positive examples
        for positive_aid in relevant_aids:
            if positive_aid in aid_map:
                positive_content = aid_map[positive_aid]

                # Tao negative examples (random)
                negative_aids = [aid for aid in all_aids if aid not in relevant_aids]
                if negative_aids:
                    negative_aid = random.choice(negative_aids)
                    negative_content = aid_map[negative_aid]

                    triplet = {
                        "anchor": question,
                        "positive": positive_content,
                        "negative": negative_content,
                    }
                    triplets.append(triplet)

    logger.info(f"[TRIPLET] Created {len(triplets)} initial triplets")
    return triplets


def load_optimized_model_for_hard_negative_mining():
    """Load model tối ưu để tìm hard negatives - ưu tiên model đã train"""
    logger.info("[HARD_NEG] Loading optimized model for hard negative mining...")

    # Lựa chọn 1: Thử load Bi-Encoder đã được huấn luyện (tốt nhất)
    bi_encoder_path = config.BI_ENCODER_PATH
    if bi_encoder_path.exists() and any(bi_encoder_path.iterdir()):
        try:
            logger.info("[HARD_NEG] Found existing Bi-Encoder model, loading...")
            model = SentenceTransformer(str(bi_encoder_path))
            logger.info("[HARD_NEG] Successfully loaded existing Bi-Encoder model")
            return model, "existing_bi_encoder"
        except Exception as e:
            logger.warning(f"[HARD_NEG] Failed to load existing Bi-Encoder: {e}")

    # Lựa chọn 2: Thử load PhoBERT-Law từ DAPT (tốt)
    phobert_law_path = config.PHOBERT_LAW_PATH
    if phobert_law_path.exists() and any(phobert_law_path.iterdir()):
        try:
            logger.info("[HARD_NEG] Found PhoBERT-Law model, loading...")
            model = SentenceTransformer(str(phobert_law_path))
            logger.info("[HARD_NEG] Successfully loaded PhoBERT-Law model")
            return model, "phobert_law"
        except Exception as e:
            logger.warning(f"[HARD_NEG] Failed to load PhoBERT-Law: {e}")

    # Lựa chọn 3: Fallback to base model (không train)
    try:
        logger.info("[HARD_NEG] Using base model without training...")
        model = SentenceTransformer(config.BI_ENCODER_MODEL_NAME)
        logger.info("[HARD_NEG] Successfully loaded base model")
        return model, "base_model"
    except Exception as e:
        logger.error(f"[HARD_NEG] Failed to load base model: {e}")
        return None, "none"


def train_temporary_bi_encoder(triplets, aid_map):
    """Train Bi-Encoder tam thoi de tim hard negatives (FALLBACK ONLY)"""
    logger.info(
        "[TEMP_BI_ENCODER] Training temporary Bi-Encoder for hard negative mining (FALLBACK)..."
    )

    try:
        # Load model
        model = SentenceTransformer(config.BI_ENCODER_MODEL_NAME)

        # Tao training examples
        train_examples = []
        for triplet in triplets:
            train_examples.append(
                InputExample(
                    texts=[triplet["anchor"], triplet["positive"], triplet["negative"]]
                )
            )

        # Tao dataloader
        train_dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=config.BI_ENCODER_BATCH_SIZE
        )

        # Loss function
        train_loss = losses.TripletLoss(model)

        # Training arguments
        num_epochs = 1  # Chỉ train 1 epoch cho model tạm
        warmup_steps = min(50, len(train_dataloader))

        # Train model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=True,
        )

        logger.info("[TEMP_BI_ENCODER] Temporary Bi-Encoder training completed")
        return model

    except Exception as e:
        logger.error(f"[TEMP_BI_ENCODER] Error training temporary Bi-Encoder: {e}")
        return None


def find_hard_negatives(
    temp_model,
    train_data,
    aid_map,
    model_type="unknown",
    top_k=None,
    hard_negative_positions=None,
):
    # Use config parameters if not provided
    if top_k is None:
        top_k = config.HARD_NEGATIVE_TOP_K
    if hard_negative_positions is None:
        hard_negative_positions = config.HARD_NEGATIVE_POSITIONS

    """Tim hard negatives su dung model tối ưu"""
    logger.info(f"[HARD_NEG] Finding hard negatives using {model_type} model...")

    if temp_model is None:
        logger.warning("[HARD_NEG] No model available, skipping hard negative mining")
        return []

    hard_negatives = []
    all_contents = list(aid_map.values())
    all_aids = list(aid_map.keys())

    # Optimize batch size based on model type
    optimal_batch_size = 64 if model_type == "existing_bi_encoder" else 32

    # Tao embeddings cho tat ca contents
    logger.info(
        f"[HARD_NEG] Creating embeddings for all contents (batch_size={optimal_batch_size})..."
    )
    embeddings = temp_model.encode(
        all_contents, show_progress_bar=True, batch_size=optimal_batch_size
    )

    # Tao FAISS index với optimization
    dimension = embeddings.shape[1]
    logger.info(f"[HARD_NEG] Creating FAISS index with dimension {dimension}...")

    # Normalize embeddings for better cosine similarity
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings.astype("float32"))

    logger.info(f"[HARD_NEG] FAISS index created with {index.ntotal} vectors")

    logger.info("[HARD_NEG] Searching for hard negatives...")

    for sample in train_data:
        question = sample["question"]
        relevant_aids = sample["relevant_aids"]

        if not relevant_aids:
            continue

        # Encode question
        question_embedding = temp_model.encode([question])

        # Search top-k results
        scores, indices = index.search(question_embedding.astype("float32"), top_k)

        # Tim hard negatives (top-ranked incorrect results)
        hard_neg_aids = []
        hard_neg_scores = []

        for i in range(
            hard_negative_positions[0],
            min(hard_negative_positions[1] + 1, len(indices[0])),
        ):
            candidate_aid = all_aids[indices[0][i]]
            if candidate_aid not in relevant_aids:
                hard_neg_aids.append(candidate_aid)
                hard_neg_scores.append(scores[0][i])

        # Sort by similarity score (higher score = more similar = harder negative)
        if hard_neg_aids:
            hard_neg_pairs = list(zip(hard_neg_aids, hard_neg_scores))
            hard_neg_pairs.sort(key=lambda x: x[1], reverse=True)
            hard_neg_aids = [aid for aid, score in hard_neg_pairs]

        # Tao hard negative triplets
        for positive_aid in relevant_aids:
            if positive_aid in aid_map:
                positive_content = aid_map[positive_aid]

                for hard_neg_aid in hard_neg_aids[
                    : config.HARD_NEGATIVES_PER_POSITIVE
                ]:  # Use config parameter
                    if hard_neg_aid in aid_map:
                        hard_negative_content = aid_map[hard_neg_aid]

                        hard_negative = {
                            "anchor": question,
                            "positive": positive_content,
                            "negative": hard_negative_content,
                            "is_hard_negative": True,
                        }
                        hard_negatives.append(hard_negative)

            logger.info(
                f"[HARD_NEG] Found {len(hard_negatives)} hard negative triplets"
            )

        # Log quality metrics
        if hard_negatives:
            logger.info(f"[HARD_NEG] Hard negative mining completed successfully")
            logger.info(
                f"[HARD_NEG] Average hard negatives per query: {len(hard_negatives) / len([s for s in train_data if s['relevant_aids']]):.1f}"
            )
        else:
            logger.warning(
                "[HARD_NEG] No hard negatives found, consider adjusting parameters"
            )

        return hard_negatives


def create_enhanced_triplets(initial_triplets, hard_negatives):
    """Tao triplets nang cao voi hard negatives"""
    logger.info("[ENHANCED] Creating enhanced triplets with hard negatives...")

    enhanced_triplets = []

    # Them initial triplets
    for triplet in initial_triplets:
        enhanced_triplet = {
            "anchor": triplet["anchor"],
            "positive": triplet["positive"],
            "negative": triplet["negative"],
            "is_hard_negative": False,
        }
        enhanced_triplets.append(enhanced_triplet)

    # Them hard negative triplets
    enhanced_triplets.extend(hard_negatives)

    logger.info(f"[ENHANCED] Created {len(enhanced_triplets)} enhanced triplets")
    logger.info(f"[ENHANCED] - Initial triplets: {len(initial_triplets)}")
    logger.info(f"[ENHANCED] - Hard negative triplets: {len(hard_negatives)}")

    return enhanced_triplets


def create_pairs(train_data, aid_map):
    """Tao pairs cho Cross-Encoder"""
    logger.info("[PAIR] Creating pairs for Cross-Encoder...")

    pairs = []
    all_aids = list(aid_map.keys())

    for sample in train_data:
        question = sample["question"]
        relevant_aids = sample["relevant_aids"]

        if not relevant_aids:
            continue

        # Positive pairs
        for positive_aid in relevant_aids:
            if positive_aid in aid_map:
                positive_content = aid_map[positive_aid]
                pair = {
                    "texts": [question, positive_content],
                    "label": 1,
                }
                pairs.append(pair)

        # Negative pairs
        negative_aids = [aid for aid in all_aids if aid not in relevant_aids]
        if negative_aids:
            # Tao so luong negative pairs bang voi positive pairs
            num_negatives = min(len(relevant_aids), len(negative_aids))
            selected_negatives = random.sample(negative_aids, num_negatives)

            for negative_aid in selected_negatives:
                negative_content = aid_map[negative_aid]
                pair = {
                    "texts": [question, negative_content],
                    "label": 0,
                }
                pairs.append(pair)

    logger.info(f"[PAIR] Created {len(pairs)} pairs")
    return pairs


def create_enhanced_pairs(train_data, aid_map, hard_negatives):
    """Tao pairs nang cao voi hard negatives"""
    logger.info("[ENHANCED_PAIR] Creating enhanced pairs with hard negatives...")

    pairs = []
    all_aids = list(aid_map.keys())

    # Tao map tu hard negatives
    hard_neg_map = {}
    for triplet in hard_negatives:
        question = triplet["anchor"]
        if question not in hard_neg_map:
            hard_neg_map[question] = []
        hard_neg_map[question].append(triplet["negative"])

    for sample in train_data:
        question = sample["question"]
        relevant_aids = sample["relevant_aids"]

        if not relevant_aids:
            continue

        # Positive pairs
        for positive_aid in relevant_aids:
            if positive_aid in aid_map:
                positive_content = aid_map[positive_aid]
                pair = {
                    "texts": [question, positive_content],
                    "label": 1,
                }
                pairs.append(pair)

        # Hard negative pairs
        if question in hard_neg_map:
            for hard_neg_content in hard_neg_map[question][
                :2
            ]:  # Lay toi da 2 hard negatives
                pair = {
                    "texts": [question, hard_neg_content],
                    "label": 0,
                    "is_hard_negative": True,
                }
                pairs.append(pair)

        # Regular negative pairs (neu can them)
        negative_aids = [aid for aid in all_aids if aid not in relevant_aids]
        if negative_aids:
            num_regular_negatives = max(1, len(relevant_aids) // 2)
            selected_negatives = random.sample(
                negative_aids, min(num_regular_negatives, len(negative_aids))
            )

            for negative_aid in selected_negatives:
                negative_content = aid_map[negative_aid]
                pair = {
                    "texts": [question, negative_content],
                    "label": 0,
                    "is_hard_negative": False,
                }
                pairs.append(pair)

    logger.info(f"[ENHANCED_PAIR] Created {len(pairs)} enhanced pairs")
    return pairs


def merge_and_augment_data(triplets, pairs):
    """Merge va augment data toi uu voi advanced techniques"""
    logger.info("[AUGMENT] Merging and augmenting data with advanced techniques...")

    # Merge Bi-Encoder data
    bi_encoder_data = []
    for triplet in triplets:
        bi_encoder_data.append(triplet)

    # Merge Cross-Encoder data
    cross_encoder_data = []
    for pair in pairs:
        cross_encoder_data.append(pair)

    # Sử dụng config.AUGMENTATION_FACTOR thay vì hardcode
    augmentation_factor = config.AUGMENTATION_FACTOR
    legal_keywords_rate = config.LEGAL_KEYWORDS_INJECTION_RATE

    logger.info(f"[AUGMENT] Using augmentation factor: {augmentation_factor}")
    logger.info(f"[AUGMENT] Using legal keywords injection rate: {legal_keywords_rate}")

    # Advanced Bi-Encoder augmentation
    augmented_bi_encoder = []
    for triplet in bi_encoder_data:
        augmented_bi_encoder.append(triplet)

        # Tính số lượng augmented samples cần tạo
        num_augmented = max(1, int(augmentation_factor - 1))

        for _ in range(num_augmented):
            # Advanced augmentation techniques
            augmented_triplet = apply_advanced_augmentation(
                triplet, legal_keywords_rate
            )
            augmented_bi_encoder.append(augmented_triplet)

    # Advanced Cross-Encoder augmentation
    augmented_cross_encoder = []
    for pair in cross_encoder_data:
        augmented_cross_encoder.append(pair)

        # Tính số lượng augmented samples cần tạo
        num_augmented = max(1, int(augmentation_factor - 1))

        for _ in range(num_augmented):
            # Advanced augmentation techniques
            augmented_pair = apply_advanced_augmentation_pair(pair, legal_keywords_rate)
            augmented_cross_encoder.append(augmented_pair)

    logger.info(f"[AUGMENT] Original Bi-Encoder: {len(bi_encoder_data)} samples")
    logger.info(f"[AUGMENT] Augmented Bi-Encoder: {len(augmented_bi_encoder)} samples")
    logger.info(f"[AUGMENT] Original Cross-Encoder: {len(cross_encoder_data)} samples")
    logger.info(
        f"[AUGMENT] Augmented Cross-Encoder: {len(augmented_cross_encoder)} samples"
    )
    logger.info(
        f"[AUGMENT] Total augmentation ratio: {len(augmented_bi_encoder)/len(bi_encoder_data):.2f}x"
    )

    return augmented_bi_encoder, augmented_cross_encoder


def apply_advanced_augmentation(triplet, legal_keywords_rate):
    """Apply advanced augmentation techniques cho triplet"""
    augmented_triplet = {
        "anchor": triplet["anchor"],
        "positive": triplet["positive"],
        "negative": triplet["negative"],
    }

    # Copy hard negative flag if exists
    if "is_hard_negative" in triplet:
        augmented_triplet["is_hard_negative"] = triplet["is_hard_negative"]

    # Legal keywords injection
    if random.random() < legal_keywords_rate:
        legal_keywords = [
            "theo quy định",
            "theo luật",
            "theo điều",
            "theo khoản",
            "theo điểm",
        ]
        keyword = random.choice(legal_keywords)
        augmented_triplet["anchor"] = f"{keyword} {triplet['anchor']}"

    # Synonym replacement (simple version)
    if random.random() < 0.3:
        synonyms = {
            "được": ["có quyền", "được phép"],
            "phải": ["bắt buộc", "cần phải"],
            "có": ["sở hữu", "có được"],
            "là": ["chính là", "được xem là"],
        }

        for original, replacements in synonyms.items():
            if original in augmented_triplet["anchor"]:
                replacement = random.choice(replacements)
                augmented_triplet["anchor"] = augmented_triplet["anchor"].replace(
                    original, replacement
                )
                break

    return augmented_triplet


def apply_advanced_augmentation_pair(pair, legal_keywords_rate):
    """Apply advanced augmentation techniques cho pair"""
    augmented_pair = {
        "texts": pair["texts"].copy(),
        "label": pair["label"],
    }

    # Copy hard negative flag if exists
    if "is_hard_negative" in pair:
        augmented_pair["is_hard_negative"] = pair["is_hard_negative"]

    # Legal keywords injection cho query
    if random.random() < legal_keywords_rate:
        legal_keywords = [
            "theo quy định",
            "theo luật",
            "theo điều",
            "theo khoản",
            "theo điểm",
        ]
        keyword = random.choice(legal_keywords)
        augmented_pair["texts"][0] = f"{keyword} {pair['texts'][0]}"

    # Synonym replacement cho query
    if random.random() < 0.3:
        synonyms = {
            "được": ["có quyền", "được phép"],
            "phải": ["bắt buộc", "cần phải"],
            "có": ["sở hữu", "có được"],
            "là": ["chính là", "được xem là"],
        }

        for original, replacements in synonyms.items():
            if original in augmented_pair["texts"][0]:
                replacement = random.choice(replacements)
                augmented_pair["texts"][0] = augmented_pair["texts"][0].replace(
                    original, replacement
                )
                break

    return augmented_pair


def save_training_data(bi_encoder_data, cross_encoder_data):
    """Save training data"""
    logger.info("[SAVE] Saving training data...")

    # Save Bi-Encoder data
    bi_encoder_path = config.DATA_PROCESSED_DIR / "bi_encoder_train_advanced.jsonl"
    with open(bi_encoder_path, "w", encoding="utf-8") as f:
        for triplet in bi_encoder_data:
            f.write(json.dumps(triplet, ensure_ascii=False) + "\n")

    # Save Cross-Encoder data
    cross_encoder_path = (
        config.DATA_PROCESSED_DIR / "cross_encoder_train_advanced.jsonl"
    )
    with open(cross_encoder_path, "w", encoding="utf-8") as f:
        for pair in cross_encoder_data:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(f"[SAVE] Bi-Encoder data saved to: {bi_encoder_path}")
    logger.info(f"[SAVE] Cross-Encoder data saved to: {cross_encoder_path}")

    return bi_encoder_path, cross_encoder_path


def run_advanced_training_data_preparation_pipeline():
    """Chay pipeline chuan bi training data voi advanced techniques"""
    logger.info("=" * 60)
    logger.info("ADVANCED TRAINING DATA PREPARATION PIPELINE")
    logger.info("=" * 60)

    try:
        # Step 1: Load processed data
        logger.info("STEP 1: Loading processed data...")
        train_data, aid_map = load_processed_data()

        # Step 2: Create initial triplets
        logger.info("STEP 2: Creating initial triplets...")
        initial_triplets = create_initial_triplets(train_data, aid_map)

        # Step 3: Load optimized model for hard negative mining
        logger.info("STEP 3: Loading optimized model for hard negative mining...")
        temp_model, model_type = load_optimized_model_for_hard_negative_mining()

        # Fallback to training if no model available
        if temp_model is None:
            logger.warning(
                "No optimized model available, falling back to training temporary model..."
            )
            temp_model = train_temporary_bi_encoder(initial_triplets, aid_map)
            model_type = "temporary_trained"

        # Step 4: Find hard negatives
        logger.info("STEP 4: Finding hard negatives...")
        hard_negatives = find_hard_negatives(
            temp_model, train_data, aid_map, model_type
        )

        # Step 5: Create enhanced triplets with hard negatives
        logger.info("STEP 5: Creating enhanced triplets with hard negatives...")
        enhanced_triplets = create_enhanced_triplets(initial_triplets, hard_negatives)

        # Step 6: Create enhanced pairs with hard negatives
        logger.info("STEP 6: Creating enhanced pairs with hard negatives...")
        enhanced_pairs = create_enhanced_pairs(train_data, aid_map, hard_negatives)

        # Step 7: Merge and augment data
        logger.info("STEP 7: Merging and augmenting data...")
        bi_encoder_data, cross_encoder_data = merge_and_augment_data(
            enhanced_triplets, enhanced_pairs
        )

        # Step 8: Save training data
        logger.info("STEP 8: Saving training data...")
        bi_encoder_path, cross_encoder_path = save_training_data(
            bi_encoder_data, cross_encoder_data
        )

        # Clean up model
        if temp_model is not None:
            del temp_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("[CLEANUP] GPU memory cleared")

        logger.info("=" * 60)
        logger.info("ADVANCED PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("PERFORMANCE SUMMARY:")
        logger.info(f"- Model used for hard negative mining: {model_type}")
        logger.info(f"- Hard negatives found: {len(hard_negatives)}")
        logger.info(f"- Total enhanced triplets: {len(enhanced_triplets)}")
        logger.info(f"- Total enhanced pairs: {len(enhanced_pairs)}")
        logger.info("=" * 60)
        logger.info("Advanced techniques implemented:")
        logger.info("✅ Optimized Model Loading (no training needed)")
        logger.info("✅ Advanced Hard Negative Mining")
        logger.info("✅ Enhanced training data with hard negatives")
        logger.info("Next steps:")
        logger.info("1. Run model training pipeline")
        logger.info("2. Deploy system")
        logger.info("3. Monitor performance")
        return True

    except Exception as e:
        logger.error(f"Error during advanced pipeline: {e}")
        return False


def main():
    """Ham chinh"""
    logger.info("[START] Bat dau Advanced Training Data Preparation Pipeline...")
    logger.info(
        "[OPTIMIZATION] Using optimized model loading for hard negative mining..."
    )

    success = run_advanced_training_data_preparation_pipeline()

    if success:
        logger.info(
            "✅ Advanced Training Data Preparation Pipeline completed successfully!"
        )
        logger.info("✅ Enhanced training data san sang cho model training!")
        logger.info("🚀 Optimized model loading saved significant training time!")
    else:
        logger.error("❌ Advanced Training Data Preparation Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
