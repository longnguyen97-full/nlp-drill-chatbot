#!/usr/bin/env python3
"""
Core Utilities Package - LawBot v8.0
====================================

Unified utilities package combining all common functions.
"""

from .data_processing import (
    parse_legal_corpus,
    validate_legal_corpus_structure,
    load_json,
    load_pickle,
    load_jsonl,
    save_json,
    save_pickle,
    save_lines,
    split_json_data,
    split_lines_data,
)

from .model_utils import (
    create_reranker_preprocess_function,
    create_light_reranker_preprocess_function,
    compute_metrics,
)

from .evaluation import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    mean_reciprocal_rank,
    sample_random_negatives,
)

from .augmentation import (
    augment_punctuation,
    augment_legal_keywords,
    augment_text,
)

from .file_utils import (
    create_directories,
    setup_script_paths,
    load_config,
)

# Export main functions for backward compatibility
__all__ = [
    # Data processing
    "parse_legal_corpus",
    "validate_legal_corpus_structure",
    "load_json",
    "load_pickle",
    "load_jsonl",
    "save_json",
    "save_pickle",
    "save_lines",
    "split_json_data",
    "split_lines_data",
    # Model utilities
    "create_reranker_preprocess_function",
    "create_light_reranker_preprocess_function",
    "compute_metrics",
    # Evaluation
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "mean_reciprocal_rank",
    "sample_random_negatives",
    # Augmentation
    "augment_punctuation",
    "augment_legal_keywords",
    "augment_text",
    # File utilities
    "create_directories",
    "setup_script_paths",
    "load_config",
]
