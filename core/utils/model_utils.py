#!/usr/bin/env python3
"""
Model Utilities - LawBot v8.0
=============================

Model-related utilities for preprocessing and evaluation with enhanced error handling.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from transformers import PreTrainedTokenizer
from core.logging_system import get_logger
import traceback
import time
from contextlib import contextmanager

logger = get_logger(__name__)


class ModelUtilsError(Exception):
    """Custom exception for model utilities"""

    pass


def validate_tokenizer(tokenizer: PreTrainedTokenizer) -> bool:
    """Validate tokenizer configuration"""
    try:
        if not hasattr(tokenizer, "encode"):
            logger.error("Tokenizer must have encode method")
            return False

        if not hasattr(tokenizer, "decode"):
            logger.error("Tokenizer must have decode method")
            return False

        # Test basic functionality
        test_text = "test"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)

        if not decoded:
            logger.error("Tokenizer decode returned empty result")
            return False

        return True

    except Exception as e:
        logger.error(f"Tokenizer validation failed: {e}")
        return False


def validate_model_inputs(examples: Dict[str, Any], required_keys: List[str]) -> bool:
    """Validate model input data structure"""
    try:
        if not isinstance(examples, dict):
            logger.error(f"Examples must be a dict, got {type(examples)}")
            return False

        for key in required_keys:
            if key not in examples:
                logger.error(f"Missing required key: {key}")
                return False

        # Check if all inputs are lists of same length
        lengths = []
        for key in required_keys:
            if isinstance(examples[key], list):
                lengths.append(len(examples[key]))
            else:
                logger.error(f"Key {key} must be a list, got {type(examples[key])}")
                return False

        if len(set(lengths)) > 1:
            logger.error(
                f"All input lists must have same length, got lengths: {lengths}"
            )
            return False

        return True

    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        return False


@contextmanager
def model_operation_timer(operation_name: str):
    """Context manager for timing model operations"""
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    try:
        yield
    finally:
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        duration = end_time - start_time
        memory_diff = end_memory - start_memory

        logger.info(
            f"[MODEL] {operation_name}: {duration:.2f}s, "
            f"Memory: {memory_diff/1024**2:.1f}MB"
        )


def create_reranker_preprocess_function(
    tokenizer: PreTrainedTokenizer, max_length: int
):
    """
    Tạo function preprocessing cho Cross-Encoder/Reranker models với enhanced error handling.

    Args:
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length

    Returns:
        Function để preprocess data
    """
    if not validate_tokenizer(tokenizer):
        raise ModelUtilsError("Invalid tokenizer provided")

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
            required_keys = ["text1", "text2", "label"]
            if not validate_model_inputs(examples, required_keys):
                raise ModelUtilsError("Invalid input structure")

            # Ensure all inputs are lists of same length
            text1_list = examples["text1"]
            text2_list = examples["text2"]
            label_list = examples["label"]

            # Create combined texts with error handling
            texts = []
            for q, p in zip(text1_list, text2_list):
                try:
                    combined_text = f"{q} [SEP] {p}"
                    texts.append(combined_text)
                except Exception as e:
                    logger.warning(f"Error combining text pair: {e}")
                    texts.append("")  # Fallback

            # Tokenize with error handling
            with model_operation_timer("Tokenization"):
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
                try:
                    if isinstance(label, (int, float)):
                        valid_labels.append(int(label))
                    else:
                        logger.warning(f"Invalid label type: {type(label)}, using 0")
                        valid_labels.append(0)
                except Exception as e:
                    logger.warning(f"Error processing label {label}: {e}")
                    valid_labels.append(0)

            result["labels"] = torch.tensor(valid_labels, dtype=torch.long)

            return result

        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            logger.error(traceback.format_exc())
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
    Tạo function preprocessing cho Light Reranker models với enhanced error handling.

    Args:
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length

    Returns:
        Function để preprocess data
    """
    if not validate_tokenizer(tokenizer):
        raise ModelUtilsError("Invalid tokenizer provided")

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
            required_keys = ["texts", "label"]
            if not validate_model_inputs(examples, required_keys):
                raise ModelUtilsError("Invalid input structure")

            # Process texts with error handling
            texts = []
            for i in range(len(examples["texts"])):
                try:
                    question = examples["texts"][i][0]
                    answer = examples["texts"][i][1]
                    combined_text = f"{question} [SEP] {answer}"
                    texts.append(combined_text)
                except Exception as e:
                    logger.warning(f"Error processing text pair {i}: {e}")
                    texts.append("")  # Fallback

            # Tokenize with error handling
            with model_operation_timer("Light Reranker Tokenization"):
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
                try:
                    if isinstance(label, (int, float)):
                        valid_labels.append(int(label))
                    else:
                        logger.warning(f"Invalid label type: {type(label)}, using 0")
                        valid_labels.append(0)
                except Exception as e:
                    logger.warning(f"Error processing label {label}: {e}")
                    valid_labels.append(0)

            result["labels"] = torch.tensor(valid_labels, dtype=torch.long)

            return result

        except Exception as e:
            logger.error(f"Light reranker preprocessing error: {e}")
            logger.error(traceback.format_exc())
            # Return empty result to avoid pipeline crash
            return {
                "input_ids": torch.tensor([]),
                "attention_mask": torch.tensor([]),
                "labels": torch.tensor([]),
            }

    return preprocess_function


def compute_metrics(pred) -> Dict[str, float]:
    """
    Compute evaluation metrics for classification tasks with enhanced error handling.

    Args:
        pred: Prediction object from trainer

    Returns:
        Dict containing accuracy and other metrics
    """
    try:
        if not hasattr(pred, "label_ids") or not hasattr(pred, "predictions"):
            logger.error("Prediction object missing required attributes")
            return {"accuracy": 0.0}

        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        # Validate data
        if labels.shape != preds.shape:
            logger.error(
                f"Shape mismatch: labels {labels.shape}, predictions {preds.shape}"
            )
            return {"accuracy": 0.0}

        # Calculate accuracy
        accuracy = (preds == labels).astype(float).mean()

        # Calculate precision, recall, f1 for binary classification
        if len(np.unique(labels)) == 2:  # Binary classification
            try:
                from sklearn.metrics import precision_recall_fscore_support

                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels, preds, average="binary"
                )

                return {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }
            except ImportError:
                logger.warning("sklearn not available, returning only accuracy")
                return {"accuracy": float(accuracy)}
        else:
            return {"accuracy": float(accuracy)}

    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        logger.error(traceback.format_exc())
        return {"accuracy": 0.0}


def validate_model_predictions(predictions: np.ndarray, labels: np.ndarray) -> bool:
    """Validate model predictions and labels"""
    try:
        if predictions is None or labels is None:
            logger.error("Predictions or labels are None")
            return False

        if not isinstance(predictions, np.ndarray) or not isinstance(
            labels, np.ndarray
        ):
            logger.error("Predictions and labels must be numpy arrays")
            return False

        if predictions.shape != labels.shape:
            logger.error(
                f"Shape mismatch: predictions {predictions.shape}, labels {labels.shape}"
            )
            return False

        if len(predictions) == 0:
            logger.error("Empty predictions")
            return False

        return True

    except Exception as e:
        logger.error(f"Prediction validation failed: {e}")
        return False


def safe_model_inference(
    model: torch.nn.Module, inputs: Dict[str, torch.Tensor], device: str = "cpu"
) -> Optional[torch.Tensor]:
    """
    Safely perform model inference with error handling.

    Args:
        model: The model to use for inference
        inputs: Input tensors
        device: Device to use for inference

    Returns:
        Model outputs or None if inference fails
    """
    try:
        with torch.no_grad():
            # Move inputs to device
            device_inputs = {k: v.to(device) for k, v in inputs.items()}

            # Perform inference
            outputs = model(**device_inputs)

            return outputs.logits if hasattr(outputs, "logits") else outputs

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"GPU out of memory during inference: {e}")
            # Try to clear cache and retry on CPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Retrying inference on CPU")
            return safe_model_inference(model, inputs, "cpu")
        else:
            logger.error(f"Runtime error during inference: {e}")
            return None

    except Exception as e:
        logger.error(f"Error during model inference: {e}")
        logger.error(traceback.format_exc())
        return None


def optimize_batch_size(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    max_memory_gb: float = 8.0,
    initial_batch_size: int = 16,
) -> int:
    """
    Dynamically optimize batch size based on available memory.

    Args:
        model: The model to test
        tokenizer: Tokenizer for creating test inputs
        max_length: Maximum sequence length
        max_memory_gb: Maximum memory usage in GB
        initial_batch_size: Initial batch size to test

    Returns:
        Optimized batch size
    """
    try:
        # Create test inputs
        test_texts = ["test text"] * initial_batch_size
        inputs = tokenizer(
            test_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Test different batch sizes
        for batch_size in [initial_batch_size, 8, 4, 2, 1]:
            try:
                # Create batch
                batch_inputs = {k: v[:batch_size] for k, v in inputs.items()}

                # Test inference
                with torch.no_grad():
                    _ = model(**batch_inputs)

                logger.info(f"Optimized batch size: {batch_size}")
                return batch_size

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"Batch size {batch_size} too large, trying smaller")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

    except Exception as e:
        logger.error(f"Error optimizing batch size: {e}")
        return 1  # Return safe default

    return 1  # Return safe default if all else fails


def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """Get comprehensive model information"""
    try:
        info = {
            "model_type": type(model).__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "device": next(model.parameters()).device,
        }

        # Add model-specific info
        if hasattr(model, "config"):
            info["config"] = {
                "hidden_size": getattr(model.config, "hidden_size", "unknown"),
                "num_layers": getattr(model.config, "num_hidden_layers", "unknown"),
                "vocab_size": getattr(model.config, "vocab_size", "unknown"),
            }

        return info

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {"error": str(e)}


def cleanup_model_resources(model: torch.nn.Module):
    """Clean up model resources and free memory"""
    try:
        # Move model to CPU to free GPU memory
        if torch.cuda.is_available():
            model.cpu()
            torch.cuda.empty_cache()

        # Delete model references
        del model

        # Force garbage collection
        import gc

        gc.collect()

        logger.info("Model resources cleaned up successfully")

    except Exception as e:
        logger.error(f"Error cleaning up model resources: {e}")
