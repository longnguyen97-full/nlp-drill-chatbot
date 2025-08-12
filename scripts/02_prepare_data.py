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
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader
import faiss
from tqdm import tqdm
import sys
import os
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import config
from core.io import save_json, load_json, load_pickle, save_pickle, parse_legal_corpus
from core.services.logging_service import get_logger
from core.progress_tracker import StepLogger

logger = get_logger(__name__)

def ensure_aid_map_exists():
    """Tạo aid_map từ corpus nếu chưa có."""
    step_logger = StepLogger("02")
    if config.AID_MAP_PATH.exists():
        step_logger.info("aid_map.pkl already exists. Skipping creation.")
        return True
    
    step_logger.warning("aid_map.pkl not found. Creating it from legal_corpus.json...")
    try:
        all_articles = parse_legal_corpus(config.LEGAL_CORPUS_PATH)
        if not all_articles:
            raise ValueError("No articles parsed from legal corpus.")
        
        aid_map = {article['aid']: article['content'] for article in all_articles}
        
        save_pickle(aid_map, config.AID_MAP_PATH)
        step_logger.success(f"Successfully created and saved aid_map.pkl with {len(aid_map)} entries.")
        return aid_map
    except Exception as e:
        step_logger.error(f"Failed to create aid_map.pkl: {e}", exc_info=True)
        return None

def load_processed_data():
    """Load train data and aid map."""
    step_logger = StepLogger("02")
    step_logger.func_start("load_processed_data")
    try:
        if not config.TRAIN_JSON_PATH.exists():
            step_logger.error(f"Train data file not found at: {config.TRAIN_JSON_PATH}")
            raise FileNotFoundError
        with open(config.TRAIN_JSON_PATH, "r", encoding="utf-8") as f:
            train_data = json.load(f)

        if not config.AID_MAP_PATH.exists():
            step_logger.error(f"AID map file not found at: {config.AID_MAP_PATH}")
            raise FileNotFoundError
        with open(config.AID_MAP_PATH, "rb") as f:
            aid_map = pickle.load(f)

        step_logger.success(f"Loaded {len(train_data)} training examples and {len(aid_map)} AIDs.")
        step_logger.func_end("load_processed_data", ok=True)
        return train_data, aid_map
    except Exception as e:
        step_logger.error(f"Failed to load processed data: {e}", exc_info=True)
        step_logger.func_end("load_processed_data", ok=False)
        return None, None

def create_initial_triplets(train_data: List[Dict], aid_map: Dict, int_to_str_id_map: Dict) -> List[InputExample]:
    """Tạo triplets dựa trên ground truth, không còn tạo ngẫu nhiên."""
    step_logger = StepLogger("02")
    step_logger.func_start("create_initial_triplets")
    triplets = []
    
    all_aids_str = list(aid_map.keys())
    if len(all_aids_str) < 2:
        step_logger.error("Not enough articles in aid_map to create training pairs.")
        return []

    for item in tqdm(train_data, desc="Creating initial triplets from ground truth"):
        question = item.get("question", "").strip()
        relevant_laws_int = item.get("relevant_laws", [])
        
        if not question or not relevant_laws_int:
            continue
            
        # Chuyển đổi ID số nguyên sang ID chuỗi ký tự
        positive_aids_str = [int_to_str_id_map.get(int_id) for int_id in relevant_laws_int]
        positive_aids_str = [aid for aid in positive_aids_str if aid and aid in aid_map]

        if not positive_aids_str:
            continue

        for pos_aid_str in positive_aids_str:
            positive_content = aid_map.get(pos_aid_str)
            if not positive_content:
                continue

            # Chọn một negative aid ngẫu nhiên
            neg_aid_str = random.choice(all_aids_str)
            while neg_aid_str in positive_aids_str:
                neg_aid_str = random.choice(all_aids_str)
            
            negative_content = aid_map.get(neg_aid_str)
            if negative_content:
                triplets.append(InputExample(texts=[question, positive_content, negative_content]))

    if not triplets:
        step_logger.warning("Created 0 initial triplets. Check alignment between train.json and corpus.")
    else:
        step_logger.success(f"Successfully created {len(triplets)} initial triplets based on ground truth.")
    
    step_logger.func_end("create_initial_triplets", ok=True)
    return triplets

def find_hard_negatives(model: SentenceTransformer, train_data: List[Dict], aid_map: Dict, int_to_str_id_map: Dict) -> Dict[str, List[str]]:
    """
    Find hard negatives, ensuring correct ID mapping.
    """
    step_logger = StepLogger("02")
    step_logger.func_start("find_hard_negatives")

    corpus_aids_str = list(aid_map.keys())
    corpus_embeddings = model.encode(
        [aid_map[aid] for aid in corpus_aids_str],
        show_progress_bar=True,
        convert_to_tensor=True
    ).cpu().numpy()
    faiss.normalize_L2(corpus_embeddings)

    index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)

    hard_negatives_map = {}
    questions = [item['question'] for item in train_data]
    question_embeddings = model.encode(
        questions,
        show_progress_bar=True,
        convert_to_tensor=True
    ).cpu().numpy()
    faiss.normalize_L2(question_embeddings)

    top_k = config.HARD_NEGATIVES_PER_POSITIVE * 5
    _, top_k_indices = index.search(question_embeddings, top_k)

    for i in range(len(questions)):
        question_id = train_data[i]['question_id']
        # Convert ground truth int IDs to string IDs for comparison
        relevant_laws_int = train_data[i].get('relevant_laws', [])
        ground_truth_aids_str = {int_to_str_id_map.get(int_id) for int_id in relevant_laws_int}
        
        found_hard_negatives = []
        for idx in top_k_indices[i]:
            retrieved_aid_str = corpus_aids_str[idx]
            if retrieved_aid_str not in ground_truth_aids_str:
                found_hard_negatives.append(retrieved_aid_str)
            if len(found_hard_negatives) >= config.HARD_NEGATIVES_PER_POSITIVE:
                break
        hard_negatives_map[question_id] = found_hard_negatives

    step_logger.success(f"Found hard negatives for {len(hard_negatives_map)} questions.")
    step_logger.func_end("find_hard_negatives", ok=True)
    return hard_negatives_map

def create_final_dataset(initial_triplets: List[InputExample], hard_negatives: Dict[str, List[str]], train_data: List[Dict], aid_map: Dict):
    """
    Combines initial triplets with hard negatives and creates datasets for 
    both Bi-Encoder and Cross-Encoder.
    """
    step_logger = StepLogger("02")
    step_logger.func_start("create_final_dataset")

    # Create a map from question text to question_id
    question_to_id_map = {item['question']: item['question_id'] for item in train_data}

    bi_encoder_data = []
    cross_encoder_data = []

    # Process initial triplets (easy examples)
    for ex in initial_triplets:
        query, positive, negative = ex.texts
        bi_encoder_data.append({"query": query, "positive": positive, "negative": negative})
        cross_encoder_data.append({"texts": [query, positive], "label": 1})
        cross_encoder_data.append({"texts": [query, negative], "label": 0})

    # Process hard negatives
    for ex in initial_triplets:
        query, positive, _ = ex.texts
        question_id = question_to_id_map.get(query)
        if question_id and question_id in hard_negatives:
            for hard_neg_aid in hard_negatives[question_id]:
                hard_negative_content = aid_map.get(hard_neg_aid)
                if hard_negative_content:
                    bi_encoder_data.append({"query": query, "positive": positive, "negative": hard_negative_content})
                    cross_encoder_data.append({"texts": [query, hard_negative_content], "label": 0})

    step_logger.success(f"Final dataset created. Bi-encoder samples: {len(bi_encoder_data)}, Cross-encoder samples: {len(cross_encoder_data)}")
    step_logger.func_end("create_final_dataset", ok=True)
    return bi_encoder_data, cross_encoder_data

def save_training_data(bi_encoder_data, cross_encoder_data):
    """Lưu dữ liệu training."""
    step_logger = StepLogger("02")
    step_logger.func_start("save_training_data")
    try:
        save_json(bi_encoder_data, config.BI_ENCODER_TRAIN_MIXED_PATH, lines=True)
        step_logger.success(f"Bi-encoder data saved to {config.BI_ENCODER_TRAIN_MIXED_PATH}")
        
        save_json(cross_encoder_data, config.CROSS_ENCODER_TRAIN_PATH, lines=True)
        step_logger.success(f"Cross-encoder data saved to {config.CROSS_ENCODER_TRAIN_PATH}")
        
        step_logger.func_end("save_training_data", ok=True)
        return True
    except Exception as e:
        step_logger.error(f"Failed to save training data: {e}", exc_info=True)
        step_logger.func_end("save_training_data", ok=False)
        return False

def run_prepare_data_pipeline():
    """Main pipeline for data preparation."""
    step_logger = StepLogger("02")
    step_logger.step_start("Data Preparation Pipeline")

    try:
        # Ensure aid_map (string_id -> content) exists
        str_id_to_content_map = ensure_aid_map_exists()
        
        # Create a new mapping from integer index to string ID
        int_to_str_id_map = {i: s_id for i, s_id in enumerate(str_id_to_content_map.keys())}

        train_data, _ = load_processed_data()
        
        if train_data is None or str_id_to_content_map is None:
            raise ValueError("Failed to load initial data.")

        # Add question_id to train_data for tracking
        for i, item in enumerate(train_data):
            item['question_id'] = f"q_{i}"

        # Chia dữ liệu train/validation
        if len(train_data) >= config.MIN_VALIDATION_SAMPLES:
            step_logger.info(f"Splitting data into train/validation with ratio: {config.VALIDATION_SPLIT_RATIO}")
            train_set, val_set = train_test_split(train_data, test_size=config.VALIDATION_SPLIT_RATIO, random_state=42)
            save_json(val_set, config.VAL_SPLIT_JSON_PATH)
            step_logger.success(f"Validation data saved to {config.VAL_SPLIT_JSON_PATH} with {len(val_set)} samples.")
            train_data = train_set # Sử dụng phần còn lại để tạo dữ liệu training
        else:
            step_logger.warning(f"Not enough data to create validation split (have {len(train_data)}, need {config.MIN_VALIDATION_SAMPLES}). Skipping.")
            save_json(train_data, config.TRAIN_SPLIT_JSON_PATH)
            save_json([], config.VAL_SPLIT_JSON_PATH)

        initial_triplets = create_initial_triplets(train_data, str_id_to_content_map, int_to_str_id_map)

        if not initial_triplets:
            step_logger.warning("No initial triplets were created. Check your input data.")
            step_logger.step_complete(additional_info="[WARNING] No triplets created.")
            return

        # Load model for hard negative mining
        step_logger.info("Loading Bi-Encoder model for hard negative mining...")
        bi_encoder_model = SentenceTransformer(config.BI_ENCODER_MODEL_NAME)
        hard_negatives = find_hard_negatives(bi_encoder_model, train_data, str_id_to_content_map, int_to_str_id_map)

        bi_encoder_data, cross_encoder_data = create_final_dataset(initial_triplets, hard_negatives, train_data, str_id_to_content_map)
        
        if not save_training_data(bi_encoder_data, cross_encoder_data):
            raise IOError("Failed to save training data.")

        step_logger.step_complete(additional_info="Pipeline finished successfully.")

    except Exception as e:
        step_logger.error(f"Data preparation pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_prepare_data_pipeline()
