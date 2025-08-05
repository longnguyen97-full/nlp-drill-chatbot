#!/usr/bin/env python3
"""
Model Utilities - LawBot v8.0
=============================

Model-related utilities for preprocessing and evaluation.
"""

import torch
import numpy as np
from typing import Dict, Any, List
from transformers import PreTrainedTokenizer
from core.logging_system import get_logger

logger = get_logger(__name__)


def create_reranker_preprocess_function(
    tokenizer: PreTrainedTokenizer, max_length: int
):
    """
    Tạo function preprocessing cho Cross-Encoder/Reranker models.

    Args:
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length

    Returns:
        Function để preprocess data
    """

    def preprocess_function(examples):
        # Combine text1 and text2 with separator
        texts = [
            f"{text1} {tokenizer.sep_token} {text2}"
            for text1, text2 in zip(examples["text1"], examples["text2"])
        ]

        # Tokenize
        result = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None,
        )

        # Add labels
        result["labels"] = examples["label"]

        return result

    return preprocess_function


def create_light_reranker_preprocess_function(
    tokenizer: PreTrainedTokenizer, max_length: int
):
    """
    Tạo function preprocessing cho Light Reranker models.

    Args:
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length

    Returns:
        Function để preprocess data
    """

    def preprocess_function(examples):
        # Combine texts with separator
        texts = [
            f"{texts[0]} {tokenizer.sep_token} {texts[1]}"
            for texts in examples["texts"]
        ]

        # Tokenize
        result = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None,
        )

        # Add labels
        result["labels"] = examples["label"]

        return result

    return preprocess_function


def compute_metrics(pred) -> Dict[str, float]:
    """
    Compute evaluation metrics for classification tasks.

    Args:
        pred: Prediction object from trainer

    Returns:
        Dict containing accuracy and other metrics
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Calculate accuracy
    accuracy = (preds == labels).astype(float).mean()

    # Calculate precision, recall, f1 for binary classification
    if len(np.unique(labels)) == 2:  # Binary classification
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    else:
        return {"accuracy": accuracy}
