#!/usr/bin/env python3
"""
Core Utilities - Module Toi Uu Duy Nhat
==========================================

Tap hop tat ca utilities can thiet cho Legal QA Pipeline
Bao gom: Data Processing, Training Utils, Evaluation Utils

Tac gia: LawBot Team
Phien ban: Final Optimized
"""

import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logging():
    """Cau hinh logging tieu chuan cho toan bo du an."""
    # Import logging utils
    from core.logging_utils import setup_unified_logging

    # Setup hoac su dung logger hien tai
    log_file = setup_unified_logging("pipeline", "INFO")
    if log_file != "existing_logger":
        logging.info(f"Log file: {log_file}")
    else:
        logging.info("Su dung logger da duoc setup truoc do")


# ============================================================================
# DATA PROCESSING UTILITIES
# ============================================================================


class DataLoader:
    """Utility class for loading various data formats"""

    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Optional[Any]:
        """Load JSON data from file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load JSON from {file_path}: {e}")
            return None

    @staticmethod
    def load_pickle(file_path: Union[str, Path]) -> Optional[Any]:
        """Load pickle data from file"""
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f"Failed to load pickle from {file_path}: {e}")
            return None

    @staticmethod
    def load_jsonl(file_path: Union[str, Path]) -> List[Dict]:
        """Load JSONL data from file"""
        data = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data
        except Exception as e:
            logging.error(f"Failed to load JSONL from {file_path}: {e}")
            return []


class DataSaver:
    """Utility class for saving various data formats"""

    @staticmethod
    def save_json(data: Any, file_path: Union[str, Path]) -> bool:
        """Save data to JSON file"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logging.error(f"Failed to save JSON to {file_path}: {e}")
            return False

    @staticmethod
    def save_pickle(data: Any, file_path: Union[str, Path]) -> bool:
        """Save data to pickle file"""
        try:
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logging.error(f"Failed to save pickle to {file_path}: {e}")
            return False

    @staticmethod
    def save_lines(lines: List[str], file_path: Union[str, Path]) -> bool:
        """Save lines to text file"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
            return True
        except Exception as e:
            logging.error(f"Failed to save lines to {file_path}: {e}")
            return False


class DataSplitter:
    """Utility class for splitting data"""

    @staticmethod
    def split_json_data(
        data: List[Dict], train_ratio: float = 0.8, random_seed: int = 42
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split JSON data into train and validation sets"""
        random.seed(random_seed)
        random.shuffle(data)
        split_idx = int(len(data) * train_ratio)
        return data[:split_idx], data[split_idx:]

    @staticmethod
    def split_lines_data(
        lines: List[str], train_ratio: float = 0.8, random_seed: int = 42
    ) -> Tuple[List[str], List[str]]:
        """Split lines data into train and validation sets"""
        random.seed(random_seed)
        random.shuffle(lines)
        split_idx = int(len(lines) * train_ratio)
        return lines[:split_idx], lines[split_idx:]


class NegativeSampler:
    """Utility class for sampling negative examples"""

    @staticmethod
    def sample_random_negatives(
        positive_aids: List[str], all_aids: List[str], num_negatives: int = 4
    ) -> List[str]:
        """Sample random negative examples"""
        available_negatives = [aid for aid in all_aids if aid not in positive_aids]
        return random.sample(
            available_negatives, min(num_negatives, len(available_negatives))
        )


class DataAugmenter:
    """Utility class for data augmentation"""

    @staticmethod
    def augment_punctuation(text: str) -> str:
        """Augment text with punctuation variations"""
        return text

    @staticmethod
    def augment_legal_keywords(text: str) -> str:
        """Augment text with legal keyword variations"""
        return text

    @classmethod
    def augment_text(cls, text: str) -> str:
        """Apply all augmentation techniques to text"""
        augmented = cls.augment_punctuation(text)
        augmented = cls.augment_legal_keywords(augmented)
        return augmented


class MetricsCalculator:
    """Utility class for calculating evaluation metrics"""

    @staticmethod
    def precision_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
        """Calculate Precision@K"""
        if not retrieved or k == 0:
            return 0.0

        relevant_set = set(relevant)
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for item in retrieved_k if item in relevant_set)
        return relevant_retrieved / len(retrieved_k)

    @staticmethod
    def recall_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
        """Calculate Recall@K"""
        if not relevant or k == 0:
            return 0.0

        relevant_set = set(relevant)
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for item in retrieved_k if item in relevant_set)
        return relevant_retrieved / len(relevant_set)

    @staticmethod
    def f1_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
        """Calculate F1@K"""
        precision = MetricsCalculator.precision_at_k(relevant, retrieved, k)
        recall = MetricsCalculator.recall_at_k(relevant, retrieved, k)

        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def mean_reciprocal_rank(
        relevant_lists: List[List[str]], retrieved_lists: List[List[str]]
    ) -> float:
        """Calculate Mean Reciprocal Rank"""
        if not relevant_lists or not retrieved_lists:
            return 0.0

        reciprocal_ranks = []
        for relevant, retrieved in zip(relevant_lists, retrieved_lists):
            rr = 0.0
            for i, item in enumerate(retrieved, 1):
                if item in relevant:
                    rr = 1.0 / i
                    break
            reciprocal_ranks.append(rr)

        return sum(reciprocal_ranks) / len(reciprocal_ranks)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def create_directories(directories: List[Union[str, Path]]) -> None:
    """Create directories if they don't exist"""
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def setup_script_paths():
    """Setup Python path for script imports"""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def load_config():
    """Load configuration module"""
    setup_script_paths()
    try:
        import config

        return config
    except ImportError as e:
        logging.error(f"Failed to import config: {e}")
        return None


# ============================================================================
# INITIALIZATION
# ============================================================================

# Auto-setup when module is imported
setup_logging()
setup_script_paths()
