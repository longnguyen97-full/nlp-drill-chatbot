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
    Tạo function preprocessing cho Cross-Encoder/Reranker models với error handling.

    Args:
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length

    Returns:
        Function để preprocess data
    """

    def preprocess_function(examples):
        """
        Preprocess function for reranker models with robust error handling.

        Args:
            examples: Dictionary containing 'text1', 'text2', and 'label' keys

        Returns:
            dict: Tokenized inputs with labels
        """
        try:
            # Validate input structure
            if not isinstance(examples, dict):
                raise ValueError(f"Examples must be a dict, got {type(examples)}")

            required_keys = ["text1", "text2", "label"]
            for key in required_keys:
                if key not in examples:
                    raise ValueError(f"Missing required key: {key}")

            # Ensure all inputs are lists of same length
            text1_list = examples["text1"]
            text2_list = examples["text2"]
            label_list = examples["label"]

            if not isinstance(text1_list, list) or not isinstance(text2_list, list):
                raise ValueError("text1 and text2 must be lists")

            if len(text1_list) != len(text2_list):
                raise ValueError("text1 and text2 must have the same length")

            # Create combined texts
            texts = [f"{q} [SEP] {p}" for q, p in zip(text1_list, text2_list)]

            # Tokenize with error handling
            result = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors=None,  # Avoid tuple index error
            )

            # Manual tensor conversion with error handling
            try:
                result = {k: torch.tensor(v) for k, v in result.items()}
            except Exception as tensor_error:
                logger.warning(f"Tensor conversion failed, using lists: {tensor_error}")
                # Keep as lists if tensor conversion fails

            # Add labels - CRITICAL FIX: This was missing!
            valid_labels = []
            for label in label_list:
                if isinstance(label, (int, float)):
                    valid_labels.append(int(label))
                else:
                    logger.warning(f"Invalid label type: {type(label)}, using 0")
                    valid_labels.append(0)

            result["labels"] = torch.tensor(valid_labels, dtype=torch.long)

            return result

        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            # Return empty result to avoid pipeline crash
            return {
                "input_ids": torch.tensor([]),
                "attention_mask": torch.tensor([]),
                "labels": torch.tensor([]),
            }

    return preprocess_function


def create_light_reranker_preprocess_function(
    tokenizer: PreTrainedTokenizer, max_length: int
):
    """
    Tạo function preprocessing cho Light Reranker models với error handling.

    Args:
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length

    Returns:
        Function để preprocess data
    """

    def preprocess_function(examples):
        """
        Preprocess function for Light Reranker with robust error handling.

        Args:
            examples: Dictionary containing 'texts' and 'label' keys

        Returns:
            dict: Tokenized inputs with labels
        """
        try:
            # Validate input structure
            if not isinstance(examples, dict):
                raise ValueError(f"Examples must be a dict, got {type(examples)}")

            required_keys = ["texts", "label"]
            for key in required_keys:
                if key not in examples:
                    raise ValueError(f"Missing required key: {key}")

            # Process texts
            texts = []
            for i in range(len(examples["texts"])):
                question = examples["texts"][i][0]
                answer = examples["texts"][i][1]
                texts.append(f"{question} [SEP] {answer}")

            # Tokenize with error handling
            result = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors=None,  # Avoid tuple index error
            )

            # Manual tensor conversion with error handling
            try:
                result = {k: torch.tensor(v) for k, v in result.items()}
            except Exception as tensor_error:
                logger.warning(f"Tensor conversion failed, using lists: {tensor_error}")
                # Keep as lists if tensor conversion fails

            # Add labels - CRITICAL FIX: This was missing!
            valid_labels = []
            for label in examples["label"]:
                if isinstance(label, (int, float)):
                    valid_labels.append(int(label))
                else:
                    logger.warning(f"Invalid label type: {type(label)}, using 0")
                    valid_labels.append(0)

            result["labels"] = torch.tensor(valid_labels, dtype=torch.long)

            return result

        except Exception as e:
            logger.error(f"Light reranker preprocessing error: {e}")
            # Return empty result to avoid pipeline crash
            return {
                "input_ids": torch.tensor([]),
                "attention_mask": torch.tensor([]),
                "labels": torch.tensor([]),
            }

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
