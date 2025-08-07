"""
Evaluation utilities for the Legal QA Pipeline.
Provides BatchEvaluator and EvaluationReporter classes for comprehensive evaluation.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy arrays and other non-serializable objects."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, "tolist"):
            return obj.tolist()
        return super().default(obj)


class BatchEvaluator:
    """Batch evaluator for processing multiple queries efficiently."""

    def __init__(self, k_values: List[int]):
        """
        Initialize the batch evaluator.

        Args:
            k_values: List of k values for evaluation metrics
        """
        self.k_values = k_values

    def evaluate_batch(
        self,
        queries: List[str],
        ground_truth_sets: List[set],
        retrieved_aids_batch: List[List[str]],
    ) -> Dict[str, float]:
        """
        Evaluate a batch of queries.

        Args:
            queries: List of query strings
            ground_truth_sets: List of ground truth sets
            retrieved_aids_batch: List of retrieved AID lists

        Returns:
            Dictionary of average metrics
        """
        from core.utils.evaluation import precision_at_k, recall_at_k, f1_at_k

        metrics = {}
        for k in self.k_values:
            precision_values = []
            recall_values = []
            f1_values = []

            for i, (ground_truth, retrieved_aids) in enumerate(
                zip(ground_truth_sets, retrieved_aids_batch)
            ):
                if not retrieved_aids:
                    continue

                precision = precision_at_k(list(ground_truth), retrieved_aids[:k], k)
                recall = recall_at_k(list(ground_truth), retrieved_aids[:k], k)
                f1 = f1_at_k(list(ground_truth), retrieved_aids[:k], k)

                precision_values.append(precision)
                recall_values.append(recall)
                f1_values.append(f1)

            # Calculate averages
            if precision_values:
                metrics[f"precision@{k}"] = sum(precision_values) / len(
                    precision_values
                )
                metrics[f"recall@{k}"] = sum(recall_values) / len(recall_values)
                metrics[f"f1@{k}"] = sum(f1_values) / len(f1_values)
            else:
                metrics[f"precision@{k}"] = 0.0
                metrics[f"recall@{k}"] = 0.0
                metrics[f"f1@{k}"] = 0.0

        return metrics


class EvaluationReporter:
    """Comprehensive evaluation report generator."""

    def __init__(self):
        """Initialize the evaluation reporter."""
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)

    def create_comprehensive_report(
        self,
        retrieval_metrics: Dict[str, float],
        reranking_metrics: Dict[str, float],
        per_query_results: List[Dict],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create a comprehensive evaluation report.

        Args:
            retrieval_metrics: Metrics from retrieval evaluation
            reranking_metrics: Metrics from reranking evaluation
            per_query_results: Detailed per-query results
            metadata: Additional metadata

        Returns:
            Comprehensive report dictionary
        """
        report = {
            "metadata": metadata,
            "summary": {
                "retrieval_metrics": retrieval_metrics,
                "reranking_metrics": reranking_metrics,
                "overall_performance": self._calculate_overall_performance(
                    retrieval_metrics, reranking_metrics
                ),
            },
            "detailed_results": {
                "per_query_analysis": per_query_results,
                "query_count": len(per_query_results),
            },
            "recommendations": self._generate_recommendations(
                retrieval_metrics, reranking_metrics
            ),
        }

        return report

    def _calculate_overall_performance(
        self, retrieval_metrics: Dict[str, float], reranking_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate overall performance indicators."""
        overall = {}

        # Average precision@1 across retrieval and reranking
        retrieval_p1 = retrieval_metrics.get("precision@1", 0.0)
        reranking_p1 = reranking_metrics.get("precision@1", 0.0)
        overall["avg_precision@1"] = (retrieval_p1 + reranking_p1) / 2

        # Average recall@10 across retrieval and reranking
        retrieval_r10 = retrieval_metrics.get("recall@10", 0.0)
        reranking_r10 = reranking_metrics.get("recall@10", 0.0)
        overall["avg_recall@10"] = (retrieval_r10 + reranking_r10) / 2

        # Improvement from retrieval to reranking
        overall["reranking_improvement"] = reranking_p1 - retrieval_p1

        return overall

    def _generate_recommendations(
        self, retrieval_metrics: Dict[str, float], reranking_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        retrieval_p1 = retrieval_metrics.get("precision@1", 0.0)
        reranking_p1 = reranking_metrics.get("precision@1", 0.0)

        if retrieval_p1 < 0.1:
            recommendations.append(
                "Retrieval performance is very low. Consider retraining the Bi-Encoder model."
            )

        if reranking_p1 < 0.1:
            recommendations.append(
                "Reranking performance is very low. Consider retraining the Cross-Encoder model."
            )

        if reranking_p1 <= retrieval_p1:
            recommendations.append(
                "Reranking is not improving results. Review Cross-Encoder training data quality."
            )

        if retrieval_p1 > 0.5 and reranking_p1 > 0.5:
            recommendations.append(
                "Both models are performing well. Consider fine-tuning for specific use cases."
            )

        if not recommendations:
            recommendations.append(
                "Performance is within acceptable ranges. Monitor for production deployment."
            )

        return recommendations

    def save_report(self, report: Dict[str, Any]) -> str:
        """
        Save the evaluation report to file.

        Args:
            report: The evaluation report to save

        Returns:
            Path to the saved report file
        """
        timestamp = report["metadata"].get("timestamp", "latest")
        # Fix: Replace invalid characters for Windows filename
        safe_timestamp = timestamp.replace(":", "-").replace("T", "_").replace(".", "_")
        report_path = self.reports_dir / f"evaluation_report_{safe_timestamp}.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        logging.info(f"[FILE] Evaluation report saved to: {report_path}")
        return str(report_path)

    def display_summary(self, report: Dict[str, Any]):
        """Display a summary of the evaluation results."""
        summary = report["summary"]
        overall = summary["overall_performance"]

        logging.info("=" * 60)
        logging.info("EVALUATION SUMMARY")
        logging.info("=" * 60)

        # Display retrieval metrics
        logging.info("RETRIEVAL METRICS:")
        for k in [1, 3, 5, 10]:
            precision = summary["retrieval_metrics"].get(f"precision@{k}", 0.0)
            recall = summary["retrieval_metrics"].get(f"recall@{k}", 0.0)
            f1 = summary["retrieval_metrics"].get(f"f1@{k}", 0.0)
            logging.info(
                f"  P@{k}: {precision:.4f}, R@{k}: {recall:.4f}, F1@{k}: {f1:.4f}"
            )

        # Display reranking metrics
        logging.info("RERANKING METRICS:")
        for k in [1, 3, 5, 10]:
            precision = summary["reranking_metrics"].get(f"precision@{k}", 0.0)
            recall = summary["reranking_metrics"].get(f"recall@{k}", 0.0)
            f1 = summary["reranking_metrics"].get(f"f1@{k}", 0.0)
            logging.info(
                f"  P@{k}: {precision:.4f}, R@{k}: {recall:.4f}, F1@{k}: {f1:.4f}"
            )

        # Display overall performance
        logging.info("OVERALL PERFORMANCE:")
        logging.info(f"  Average Precision@1: {overall['avg_precision@1']:.4f}")
        logging.info(f"  Average Recall@10: {overall['avg_recall@10']:.4f}")
        logging.info(f"  Reranking Improvement: {overall['reranking_improvement']:.4f}")

        # Display recommendations
        logging.info("RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            logging.info(f"  {i}. {rec}")

        logging.info("=" * 60)
