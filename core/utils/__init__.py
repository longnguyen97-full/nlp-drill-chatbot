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


# Add missing classes
class StepLogger:
    """Simple step logger for pipeline steps"""

    def __init__(self, step_id):
        self.step_id = step_id
        from core.logging_system import get_logger

        self.logger = get_logger(f"step_{step_id}")

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)


def create_summary_report(steps_info, total_time):
    """Create a summary report for the pipeline"""
    from datetime import datetime

    report = []
    report.append("=" * 80)
    report.append("[CHART] BAO CAO TONG KET PIPELINE")
    report.append("=" * 80)

    for step_info in steps_info:
        status = "[OK]" if step_info["success"] else "[FAIL]"
        time_str = f"{step_info['time']:.1f}s"
        report.append(f"{status} {step_info['name']} ({time_str})")

    report.append("")
    report.append("[CHART] Thong ke:")
    successful = sum(1 for s in steps_info if s["success"])
    failed = len(steps_info) - successful
    report.append(f"   [OK] Thanh cong: {successful}")
    report.append(f"   [FAIL] That bai: {failed}")
    report.append(f"   [TIME] Tong thoi gian: {total_time/60:.1f} phut")
    report.append(f"   [CHART] Ty le thanh cong: {successful/len(steps_info)*100:.1f}%")
    report.append("=" * 80)

    return "\n".join(report)


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
    # Pipeline utilities
    "StepLogger",
    "create_summary_report",
]
