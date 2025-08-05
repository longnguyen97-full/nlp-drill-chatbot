#!/usr/bin/env python3
"""
Evaluation Utilities - LawBot v8.0
==================================

Evaluation metrics and utilities for model assessment.
"""

import random
from typing import List, Dict
from core.logging_system import get_logger

logger = get_logger(__name__)


def precision_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    """Calculate Precision@K"""
    if not retrieved or k == 0:
        return 0.0

    relevant_set = set(relevant)
    retrieved_k = retrieved[:k]
    relevant_retrieved = sum(1 for item in retrieved_k if item in relevant_set)
    return relevant_retrieved / len(retrieved_k)


def recall_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    """Calculate Recall@K"""
    if not relevant or k == 0:
        return 0.0

    relevant_set = set(relevant)
    retrieved_k = retrieved[:k]
    relevant_retrieved = sum(1 for item in retrieved_k if item in relevant_set)
    return relevant_retrieved / len(relevant_set)


def f1_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    """Calculate F1@K"""
    precision = precision_at_k(relevant, retrieved, k)
    recall = recall_at_k(relevant, retrieved, k)

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


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


def sample_random_negatives(
    positive_aids: List[str], all_aids: List[str], num_negatives: int = 4
) -> List[str]:
    """Sample random negative examples"""
    available_negatives = [aid for aid in all_aids if aid not in positive_aids]
    return random.sample(
        available_negatives, min(num_negatives, len(available_negatives))
    )
