"""
Evaluation utilities for the Legal QA Pipeline.
Provides BatchEvaluator and EvaluationReporter classes for comprehensive evaluation.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import time
from datetime import datetime
import traceback


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
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class BatchEvaluator:
    """Batch evaluator for processing multiple queries efficiently."""

    def __init__(self, k_values: List[int]):
        """
        Initialize the batch evaluator.

        Args:
            k_values: List of k values for evaluation metrics
        """
        self.k_values = sorted(k_values)  # Ensure sorted for consistency
        self.logger = logging.getLogger(__name__)

    def validate_inputs(
        self,
        queries: List[str],
        ground_truth_sets: List[set],
        retrieved_aids_batch: List[List[str]],
    ) -> bool:
        """Validate input data for evaluation"""
        if not queries or not ground_truth_sets or not retrieved_aids_batch:
            self.logger.error("Empty input data provided")
            return False

        if len(queries) != len(ground_truth_sets) or len(queries) != len(
            retrieved_aids_batch
        ):
            self.logger.error("Input lists have different lengths")
            return False

        if not all(isinstance(gt, set) for gt in ground_truth_sets):
            self.logger.error("Ground truth must be sets")
            return False

        return True

    def evaluate_batch(
        self,
        queries: List[str],
        ground_truth_sets: List[set],
        retrieved_aids_batch: List[List[str]],
    ) -> Dict[str, float]:
        """
        Evaluate a batch of queries with enhanced error handling.

        Args:
            queries: List of query strings
            ground_truth_sets: List of ground truth sets
            retrieved_aids_batch: List of retrieved AID lists

        Returns:
            Dictionary of average metrics
        """
        # Validate inputs
        if not self.validate_inputs(queries, ground_truth_sets, retrieved_aids_batch):
            return {}

        from core.utils.evaluation import precision_at_k, recall_at_k, f1_at_k

        metrics = {}
        start_time = time.time()

        try:
            for k in self.k_values:
                precision_values = []
                recall_values = []
                f1_values = []

                for i, (ground_truth, retrieved_aids) in enumerate(
                    zip(ground_truth_sets, retrieved_aids_batch)
                ):
                    try:
                        if not retrieved_aids:
                            continue

                        precision = precision_at_k(
                            list(ground_truth), retrieved_aids[:k], k
                        )
                        recall = recall_at_k(list(ground_truth), retrieved_aids[:k], k)
                        f1 = f1_at_k(list(ground_truth), retrieved_aids[:k], k)

                        precision_values.append(precision)
                        recall_values.append(recall)
                        f1_values.append(f1)

                    except Exception as e:
                        self.logger.warning(f"Error evaluating query {i}: {e}")
                        continue

                # Calculate averages with error handling
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

            evaluation_time = time.time() - start_time
            self.logger.info(f"Batch evaluation completed in {evaluation_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Error in batch evaluation: {e}")
            self.logger.error(traceback.format_exc())
            return {}

        return metrics


class EvaluationReporter:
    """Comprehensive evaluation report generator with enhanced features."""

    def __init__(self, reports_dir: Optional[Path] = None):
        """
        Initialize the evaluation reporter.

        Args:
            reports_dir: Directory to save reports (default: reports/)
        """
        self.reports_dir = reports_dir or Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def validate_metrics(self, metrics: Dict[str, float]) -> bool:
        """Validate metrics data"""
        if not isinstance(metrics, dict):
            self.logger.error("Metrics must be a dictionary")
            return False

        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                self.logger.error(f"Invalid metric value for {key}: {value}")
                return False
            if not 0 <= value <= 1:
                self.logger.warning(
                    f"Metric {key} value {value} is outside [0,1] range"
                )

        return True

    def create_comprehensive_report(
        self,
        retrieval_metrics: Dict[str, float],
        reranking_metrics: Dict[str, float],
        per_query_results: List[Dict],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create a comprehensive evaluation report with validation.

        Args:
            retrieval_metrics: Metrics from retrieval evaluation
            reranking_metrics: Metrics from reranking evaluation
            per_query_results: Detailed per-query results
            metadata: Additional metadata

        Returns:
            Comprehensive report dictionary
        """
        # Validate inputs
        if not self.validate_metrics(retrieval_metrics):
            self.logger.error("Invalid retrieval metrics")
            return {}

        if not self.validate_metrics(reranking_metrics):
            self.logger.error("Invalid reranking metrics")
            return {}

        if not isinstance(per_query_results, list):
            self.logger.error("Per query results must be a list")
            return {}

        try:
            report = {
                "metadata": {
                    **metadata,
                    "timestamp": datetime.now().isoformat(),
                    "report_version": "2.0",
                },
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
                    "successful_queries": len(
                        [r for r in per_query_results if r.get("success", False)]
                    ),
                },
                "recommendations": self._generate_recommendations(
                    retrieval_metrics, reranking_metrics
                ),
                "performance_analysis": self._analyze_performance(
                    retrieval_metrics, reranking_metrics
                ),
            }

            return report

        except Exception as e:
            self.logger.error(f"Error creating comprehensive report: {e}")
            self.logger.error(traceback.format_exc())
            return {}

    def _calculate_overall_performance(
        self, retrieval_metrics: Dict[str, float], reranking_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate overall performance indicators with error handling."""
        overall = {}

        try:
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

            # Additional metrics
            overall["retrieval_quality"] = retrieval_metrics.get("f1@5", 0.0)
            overall["reranking_quality"] = reranking_metrics.get("f1@5", 0.0)

        except Exception as e:
            self.logger.error(f"Error calculating overall performance: {e}")
            overall = {
                "avg_precision@1": 0.0,
                "avg_recall@10": 0.0,
                "reranking_improvement": 0.0,
                "retrieval_quality": 0.0,
                "reranking_quality": 0.0,
            }

        return overall

    def _analyze_performance(
        self, retrieval_metrics: Dict[str, float], reranking_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze performance patterns and trends."""
        analysis = {
            "performance_level": "unknown",
            "strengths": [],
            "weaknesses": [],
            "improvement_areas": [],
        }

        try:
            # Determine performance level
            avg_precision = (
                retrieval_metrics.get("precision@1", 0.0)
                + reranking_metrics.get("precision@1", 0.0)
            ) / 2

            if avg_precision >= 0.8:
                analysis["performance_level"] = "excellent"
            elif avg_precision >= 0.6:
                analysis["performance_level"] = "good"
            elif avg_precision >= 0.4:
                analysis["performance_level"] = "fair"
            else:
                analysis["performance_level"] = "poor"

            # Identify strengths and weaknesses
            if retrieval_metrics.get("precision@1", 0.0) > 0.7:
                analysis["strengths"].append("Strong retrieval performance")
            if reranking_metrics.get("precision@1", 0.0) > retrieval_metrics.get(
                "precision@1", 0.0
            ):
                analysis["strengths"].append("Effective reranking improvement")

            if retrieval_metrics.get("precision@1", 0.0) < 0.3:
                analysis["weaknesses"].append("Poor retrieval performance")
            if reranking_metrics.get("precision@1", 0.0) <= retrieval_metrics.get(
                "precision@1", 0.0
            ):
                analysis["weaknesses"].append("Reranking not improving results")

            # Suggest improvements
            if retrieval_metrics.get("precision@1", 0.0) < 0.5:
                analysis["improvement_areas"].append("Retrain Bi-Encoder model")
            if reranking_metrics.get("precision@1", 0.0) < 0.5:
                analysis["improvement_areas"].append("Retrain Cross-Encoder model")

        except Exception as e:
            self.logger.error(f"Error analyzing performance: {e}")

        return analysis

    def _generate_recommendations(
        self, retrieval_metrics: Dict[str, float], reranking_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        try:
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

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to error")

        return recommendations

    def save_report(self, report: Dict[str, Any]) -> str:
        """
        Save the evaluation report to file with enhanced error handling.

        Args:
            report: The evaluation report to save

        Returns:
            Path to the saved report file
        """
        try:
            if not report:
                self.logger.error("Cannot save empty report")
                return ""

            timestamp = report.get("metadata", {}).get("timestamp", "latest")
            # Fix: Replace invalid characters for Windows filename
            safe_timestamp = (
                timestamp.replace(":", "-").replace("T", "_").replace(".", "_")
            )
            report_path = self.reports_dir / f"evaluation_report_{safe_timestamp}.json"

            # Create backup of existing file if it exists
            if report_path.exists():
                backup_path = report_path.with_suffix(".json.backup")
                report_path.rename(backup_path)
                self.logger.info(f"Backed up existing report to: {backup_path}")

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

            self.logger.info(f"[FILE] Evaluation report saved to: {report_path}")
            return str(report_path)

        except Exception as e:
            self.logger.error(f"Error saving report: {e}")
            self.logger.error(traceback.format_exc())
            return ""

    def display_summary(self, report: Dict[str, Any]) -> None:
        """Display a summary of the evaluation results with enhanced formatting."""
        if not report:
            self.logger.error("Cannot display empty report")
            return

        try:
            summary = report.get("summary", {})
            overall = summary.get("overall_performance", {})
            detailed = report.get("detailed_results", {})

            self.logger.info("=" * 80)
            self.logger.info("EVALUATION SUMMARY")
            self.logger.info("=" * 80)

            # Display retrieval metrics
            self.logger.info("RETRIEVAL METRICS:")
            retrieval_metrics = summary.get("retrieval_metrics", {})
            for k in [1, 3, 5, 10]:
                precision = retrieval_metrics.get(f"precision@{k}", 0.0)
                recall = retrieval_metrics.get(f"recall@{k}", 0.0)
                f1 = retrieval_metrics.get(f"f1@{k}", 0.0)
                self.logger.info(
                    f"  P@{k}: {precision:.4f}, R@{k}: {recall:.4f}, F1@{k}: {f1:.4f}"
                )

            # Display reranking metrics
            self.logger.info("RERANKING METRICS:")
            reranking_metrics = summary.get("reranking_metrics", {})
            for k in [1, 3, 5, 10]:
                precision = reranking_metrics.get(f"precision@{k}", 0.0)
                recall = reranking_metrics.get(f"recall@{k}", 0.0)
                f1 = reranking_metrics.get(f"f1@{k}", 0.0)
                self.logger.info(
                    f"  P@{k}: {precision:.4f}, R@{k}: {recall:.4f}, F1@{k}: {f1:.4f}"
                )

            # Display overall performance
            self.logger.info("OVERALL PERFORMANCE:")
            self.logger.info(
                f"  Average Precision@1: {overall.get('avg_precision@1', 0.0):.4f}"
            )
            self.logger.info(
                f"  Average Recall@10: {overall.get('avg_recall@10', 0.0):.4f}"
            )
            self.logger.info(
                f"  Reranking Improvement: {overall.get('reranking_improvement', 0.0):.4f}"
            )

            # Display detailed statistics
            self.logger.info("DETAILED STATISTICS:")
            self.logger.info(f"  Total Queries: {detailed.get('query_count', 0)}")
            self.logger.info(
                f"  Successful Queries: {detailed.get('successful_queries', 0)}"
            )

            # Display recommendations
            recommendations = report.get("recommendations", [])
            if recommendations:
                self.logger.info("RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations, 1):
                    self.logger.info(f"  {i}. {rec}")

            # Display performance analysis
            analysis = report.get("performance_analysis", {})
            if analysis:
                self.logger.info("PERFORMANCE ANALYSIS:")
                self.logger.info(
                    f"  Performance Level: {analysis.get('performance_level', 'unknown')}"
                )
                if analysis.get("strengths"):
                    self.logger.info("  Strengths:")
                    for strength in analysis["strengths"]:
                        self.logger.info(f"    ✓ {strength}")
                if analysis.get("weaknesses"):
                    self.logger.info("  Weaknesses:")
                    for weakness in analysis["weaknesses"]:
                        self.logger.info(f"    ✗ {weakness}")

            self.logger.info("=" * 80)

        except Exception as e:
            self.logger.error(f"Error displaying summary: {e}")
            self.logger.error(traceback.format_exc())

    def export_to_csv(
        self, report: Dict[str, Any], output_path: Optional[Path] = None
    ) -> str:
        """
        Export evaluation results to CSV format.

        Args:
            report: The evaluation report
            output_path: Output CSV file path

        Returns:
            Path to the saved CSV file
        """
        try:
            import csv

            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = self.reports_dir / f"evaluation_results_{timestamp}.csv"

            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                writer.writerow(["Metric", "Retrieval", "Reranking", "Overall"])

                # Write metrics
                summary = report.get("summary", {})
                retrieval_metrics = summary.get("retrieval_metrics", {})
                reranking_metrics = summary.get("reranking_metrics", {})
                overall = summary.get("overall_performance", {})

                for k in [1, 3, 5, 10]:
                    writer.writerow(
                        [
                            f"Precision@{k}",
                            f"{retrieval_metrics.get(f'precision@{k}', 0.0):.4f}",
                            f"{reranking_metrics.get(f'precision@{k}', 0.0):.4f}",
                            "",
                        ]
                    )
                    writer.writerow(
                        [
                            f"Recall@{k}",
                            f"{retrieval_metrics.get(f'recall@{k}', 0.0):.4f}",
                            f"{reranking_metrics.get(f'recall@{k}', 0.0):.4f}",
                            "",
                        ]
                    )
                    writer.writerow(
                        [
                            f"F1@{k}",
                            f"{retrieval_metrics.get(f'f1@{k}', 0.0):.4f}",
                            f"{reranking_metrics.get(f'f1@{k}', 0.0):.4f}",
                            "",
                        ]
                    )

            self.logger.info(f"CSV report exported to: {output_path}")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            return ""
