#!/usr/bin/env python3
"""
Enhanced Model Evaluation Script
================================

Script này chạy đánh giá toàn diện với các cải thiện:
- Fixed evaluation metrics
- Reduced token overflow warnings  
- Better error handling for index errors
- Detailed per-tier performance analysis

Tác giả: LawBot Team
Phiên bản: Enhanced Evaluation v2.0
"""

import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

# Early CLI pre-parse to allow --mode fast|quality BEFORE importing config
def _apply_mode_from_cli_early():
    try:
        selected_mode = None
        for i, arg in enumerate(sys.argv[1:], start=1):
            if arg == "--mode" and i + 1 < len(sys.argv):
                selected_mode = sys.argv[i + 1]
                break
            if arg.startswith("--mode="):
                selected_mode = arg.split("=", 1)[1]
                break
        if selected_mode:
            selected_mode = selected_mode.strip().lower()
            if selected_mode in ("fast", "quality"):
                os.environ["LAWBOT_PERFORMANCE_MODE"] = selected_mode
    except Exception:
        pass

_apply_mode_from_cli_early()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from core.logging_system import get_logger
from core.pipeline import LegalQAPipeline
from core.evaluation_reporter import EvaluationReporter, BatchEvaluator

# Setup logging
logger = get_logger(__name__)


def run_enhanced_evaluation():
    """Chạy đánh giá toàn diện với các cải thiện"""
    logger.info("=" * 80)
    logger.info("ENHANCED MODEL EVALUATION v2.0")
    logger.info("=" * 80)

    try:
        # 1. Khởi tạo pipeline với enhanced error handling
        logger.info("[EVAL] Initializing enhanced pipeline...")
        pipeline = LegalQAPipeline(use_ensemble=True, use_cascaded_reranking=True)
        if not pipeline.is_ready:
            logger.error("[EVAL] Pipeline is not ready. Cannot run evaluation.")
            return False

        # 2. Load validation data
        logger.info("[EVAL] Loading validation data...")
        if not config.VAL_SPLIT_JSON_PATH.exists():
            logger.error(
                f"[EVAL] Validation data not found at {config.VAL_SPLIT_JSON_PATH}"
            )
            return False

        with open(config.VAL_SPLIT_JSON_PATH, "r", encoding="utf-8") as f:
            val_data = json.load(f)

        queries = [item["question"] for item in val_data]
        ground_truth_sets = [set(item["relevant_aids"]) for item in val_data]

        # Sanity check: Coverage of ground-truth AIDs in current index
        try:
            index_to_aid_path = config.INDEX_TO_AID_PATH
            if index_to_aid_path.exists():
                with open(index_to_aid_path, "r", encoding="utf-8") as f:
                    index_aids = set(json.load(f))
                total_gt = sum(len(s) for s in ground_truth_sets)
                present = sum(
                    1 for s in ground_truth_sets for aid in s if aid in index_aids
                )
                coverage = (present / total_gt * 100) if total_gt else 0.0
                logger.info(
                    f"[EVAL] Ground-truth coverage in index: {present}/{total_gt} ({coverage:.2f}%)"
                )
                if coverage == 0.0:
                    logger.warning(
                        "[EVAL] No ground-truth AIDs found in index. Check corpus/index alignment and AID formats."
                    )
            else:
                logger.warning(
                    f"[EVAL] INDEX_TO_AID file not found at {index_to_aid_path}; cannot verify coverage"
                )
        except Exception as e:
            logger.warning(f"[EVAL] Coverage check failed: {e}")

        logger.info(f"[EVAL] Loaded {len(queries)} validation queries")

        # 3. Đánh giá từng tầng riêng biệt
        logger.info("[EVAL] Starting tier-by-tier evaluation...")

        # Tier 1: Bi-Encoder Retrieval
        logger.info("[EVAL] === TIER 1: Bi-Encoder Retrieval ===")
        tier1_results = evaluate_tier1_retrieval(pipeline, queries, ground_truth_sets)

        # Tier 3: Cross-Encoder Reranking
        logger.info("[EVAL] === TIER 3: Cross-Encoder Reranking ===")
        tier3_results = evaluate_tier3_reranking(pipeline, queries, ground_truth_sets)

        # 4. Xây per-query results phù hợp reporter
        logger.info("[EVAL] Building per-query detailed results...")
        per_query_results = []
        for i, (query, gt_set) in enumerate(zip(queries, ground_truth_sets)):
            try:
                ret_aids = tier1_results["predictions"][i] if i < len(tier1_results["predictions"]) else []
                rerank_results = tier3_results["predictions"][i] if i < len(tier3_results["predictions"]) else []
                ret_scores = []
                rerank_scores = [res.get("rerank_score", 0.0) for res in rerank_results]

                # Per-query quick metrics
                ret_precision = (len(set(ret_aids) & gt_set) / len(ret_aids)) if ret_aids else 0.0
                ret_recall = (len(set(ret_aids) & gt_set) / len(gt_set)) if gt_set else 0.0
                ret_f1 = (2 * ret_precision * ret_recall / (ret_precision + ret_recall)) if (ret_precision + ret_recall) > 0 else 0.0

                rerank_aids = [res.get("aid") for res in rerank_results]
                rerank_precision = (len(set(rerank_aids) & gt_set) / len(rerank_aids)) if rerank_aids else 0.0
                rerank_recall = (len(set(rerank_aids) & gt_set) / len(gt_set)) if gt_set else 0.0
                rerank_f1 = (2 * rerank_precision * rerank_recall / (rerank_precision + rerank_recall)) if (rerank_precision + rerank_recall) > 0 else 0.0

                per_query_results.append(
                    {
                        "query_id": i,
                        "query": query,
                        "ground_truth": list(gt_set),
                        "ground_truth_count": len(gt_set),
                        "retrieval_results": {
                            "aids": ret_aids[:10],
                            "scores": ret_scores[:10],
                            "precision": ret_precision,
                            "recall": ret_recall,
                            "f1": ret_f1,
                            "found_relevant": len(set(ret_aids) & gt_set),
                        },
                        "reranking_results": {
                            "aids": rerank_aids,
                            "scores": rerank_scores,
                            "precision": rerank_precision,
                            "recall": rerank_recall,
                            "f1": rerank_f1,
                            "found_relevant": len(set(rerank_aids) & gt_set),
                        },
                        "improvement": {
                            "precision_improvement": rerank_precision - ret_precision,
                            "recall_improvement": rerank_recall - ret_recall,
                            "f1_improvement": rerank_f1 - ret_f1,
                        },
                        "success": True,
                    }
                )
            except Exception as e:
                logger.error(f"[EVAL] Error building per-query result for {i}: {e}")

        # 5. Sinh báo cáo bằng EvaluationReporter
        logger.info("[EVAL] Generating report via EvaluationReporter...")
        reporter = EvaluationReporter()
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "enhanced_comprehensive",
            "total_queries": len(queries),
        }
        report = reporter.create_comprehensive_report(
            retrieval_metrics=tier1_results["metrics"],
            reranking_metrics=tier3_results["metrics"],
            per_query_results=per_query_results,
            metadata=metadata,
        )

        reporter.display_summary(report)
        reporter.save_report(report)

        logger.info("=" * 80)
        logger.info("ENHANCED EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        return True

    except Exception as e:
        logger.error(f"[EVAL] Enhanced evaluation failed: {e}")
        logger.error(traceback.format_exc())
        return False


def evaluate_tier1_retrieval(
    pipeline: LegalQAPipeline, queries: List[str], ground_truth_sets: List[set]
) -> Dict[str, Any]:
    """Đánh giá Tầng 1 - Bi-Encoder Retrieval"""
    logger.info("[TIER1] Evaluating Bi-Encoder retrieval performance...")

    retrieval_predictions = []
    retrieval_scores = []
    tier1_metrics = {}

    try:
        for i, q in enumerate(queries):
            if i % 10 == 0:
                logger.info(f"[TIER1] Processing query {i+1}/{len(queries)}")

            try:
                retrieved_aids, distances = pipeline.retrieve(q, config.TOP_K_RETRIEVAL)
                retrieval_predictions.append(retrieved_aids)

                # Convert distances to similarity scores
                max_dist = max(distances) if distances else 1.0
                scores = [
                    1.0 - (d / max_dist) if max_dist > 0 else 0.0 for d in distances
                ]
                retrieval_scores.append(scores)

            except Exception as e:
                logger.error(f"[TIER1] Error in retrieval for query {i}: {e}")
                retrieval_predictions.append([])
                retrieval_scores.append([])

        # Calculate metrics
        evaluator = BatchEvaluator(k_values=[1, 3, 5, 10, 20, 50])
        retrieval_aids_batch = [
            [aid for aid in preds] for preds in retrieval_predictions
        ]

        try:
            computed = (
                evaluator.evaluate_batch(queries, ground_truth_sets, retrieval_aids_batch)
                if retrieval_aids_batch
                else {}
            )
        except Exception as e:
            logger.error(f"[TIER1] Error computing metrics: {e}")
            computed = {}

        if not computed:
            logger.warning("[TIER1] Empty retrieval results, using default metrics")
            tier1_metrics = {f"precision@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]}
            tier1_metrics.update({f"recall@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]})
            tier1_metrics.update({f"f1@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]})
        else:
            tier1_metrics = computed

        return {
            "predictions": retrieval_predictions,
            "scores": retrieval_scores,
            "metrics": tier1_metrics,
            "success": True,
        }

    except Exception as e:
        logger.error(f"[TIER1] Tier 1 evaluation failed: {e}")
        return {
            "predictions": [],
            "scores": [],
            "metrics": {
                **{f"precision@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]},
                **{f"recall@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]},
                **{f"f1@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]},
            },
            "success": False,
            "error": str(e),
        }


def evaluate_tier3_reranking(
    pipeline: LegalQAPipeline, queries: List[str], ground_truth_sets: List[set]
) -> Dict[str, Any]:
    """Đánh giá Tầng 3 - Cross-Encoder Reranking"""
    logger.info("[TIER3] Evaluating Cross-Encoder reranking performance...")

    tier3_predictions = []
    tier3_scores = []
    tier3_metrics = {}

    try:
        for i, q in enumerate(queries):
            if i % 10 == 0:
                logger.info(f"[TIER3] Processing query {i+1}/{len(queries)}")

            try:
                results = pipeline.predict(
                    q, top_k_retrieval=config.TOP_K_RETRIEVAL, top_k_final=10
                )
                tier3_predictions.append(results)
                scores = [res.get("rerank_score", 0.0) for res in results]
                tier3_scores.append(scores)

            except Exception as e:
                logger.error(f"[TIER3] Error in reranking for query {i}: {e}")
                tier3_predictions.append([])
                tier3_scores.append([])

        # Calculate metrics
        evaluator = BatchEvaluator(k_values=[1, 3, 5, 10, 20, 50])
        tier3_aids_batch = [
            [res["aid"] for res in preds] for preds in tier3_predictions
        ]

        try:
            computed_r = (
                evaluator.evaluate_batch(queries, ground_truth_sets, tier3_aids_batch)
                if tier3_aids_batch
                else {}
            )
        except Exception as e:
            logger.error(f"[TIER3] Error computing metrics: {e}")
            computed_r = {}

        if not computed_r:
            logger.warning("[TIER3] Empty reranking results, using default metrics")
            tier3_metrics = {f"precision@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]}
            tier3_metrics.update({f"recall@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]})
            tier3_metrics.update({f"f1@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]})
        else:
            tier3_metrics = computed_r

        return {
            "predictions": tier3_predictions,
            "scores": tier3_scores,
            "metrics": tier3_metrics,
            "success": True,
        }

    except Exception as e:
        logger.error(f"[TIER3] Tier 3 evaluation failed: {e}")
        return {
            "predictions": [],
            "scores": [],
            "metrics": {
                **{f"precision@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]},
                **{f"recall@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]},
                **{f"f1@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]},
            },
            "success": False,
            "error": str(e),
        }


def create_enhanced_report(
    tier1_results: Dict[str, Any],
    tier3_results: Dict[str, Any],
    queries: List[str],
    ground_truth_sets: List[set],
) -> Dict[str, Any]:
    """Tạo báo cáo tổng hợp với phân tích chi tiết từng tầng"""

    # Calculate overall performance
    overall_performance = calculate_overall_performance(tier1_results, tier3_results)

    # Create per-query analysis
    per_query_analysis = create_per_query_analysis(
        tier1_results, tier3_results, queries, ground_truth_sets
    )

    # Metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "enhanced_comprehensive",
        "total_queries": len(queries),
        "pipeline_config": {
            "top_k_retrieval": config.TOP_K_RETRIEVAL,
            "top_k_final": 10,
            "use_ensemble": True,
            "use_cascaded_reranking": True,
        },
        "model_paths": {
            "bi_encoder": str(config.BI_ENCODER_PATH),
            "cross_encoder": str(config.CROSS_ENCODER_PATH),
            "light_reranker": str(config.LIGHT_RERANKER_PATH),
            "faiss_index": str(config.FAISS_INDEX_PATH),
        },
        "summary_stats": {
            "avg_ground_truth_per_query": sum(len(gt) for gt in ground_truth_sets)
            / len(ground_truth_sets),
            "queries_with_ground_truth": len(
                [gt for gt in ground_truth_sets if len(gt) > 0]
            ),
            "avg_retrieval_candidates": sum(
                len(tier1_results["predictions"][i]) for i in range(len(queries))
            )
            / len(queries),
            "avg_reranking_candidates": sum(
                len(tier3_results["predictions"][i]) for i in range(len(queries))
            )
            / len(queries),
        },
        "report_version": "2.0",
        "enhancements": [
            "Fixed evaluation metrics calculation",
            "Reduced token overflow warnings",
            "Better error handling for index errors",
            "Detailed per-tier performance analysis",
        ],
    }

    return {
        "metadata": metadata,
        "summary": {
            "tier1_metrics": tier1_results["metrics"],
            "tier3_metrics": tier3_results["metrics"],
            "overall_performance": overall_performance,
        },
        "detailed_results": {
            "per_query_analysis": per_query_analysis,
            "query_count": len(queries),
            "successful_queries": len(
                [r for r in per_query_analysis if r.get("success", False)]
            ),
        },
        "tier_analysis": {
            "tier1_success": tier1_results["success"],
            "tier3_success": tier3_results["success"],
            "tier1_error": tier1_results.get("error"),
            "tier3_error": tier3_results.get("error"),
        },
    }


def calculate_overall_performance(
    tier1_results: Dict[str, Any], tier3_results: Dict[str, Any]
) -> Dict[str, float]:
    """Tính toán hiệu suất tổng thể (dựa trên flat metrics)."""
    try:
        tier1 = tier1_results["metrics"]
        tier3 = tier3_results["metrics"]

        # Averages/representative values
        avg_precision_1 = tier3.get("precision@1", 0.0)
        avg_recall_10 = tier3.get("recall@10", 0.0)

        # Improvements (Tier3 - Tier1)
        precision_improvement = tier3.get("precision@1", 0.0) - tier1.get(
            "precision@1", 0.0
        )
        recall_improvement = tier3.get("recall@10", 0.0) - tier1.get(
            "recall@10", 0.0
        )
        f1_improvement = tier3.get("f1@1", 0.0) - tier1.get("f1@1", 0.0)

        return {
            "avg_precision@1": avg_precision_1,
            "avg_recall@10": avg_recall_10,
            "reranking_improvement": precision_improvement,
            "retrieval_quality": tier1.get("f1@5", 0.0),
            "reranking_quality": tier3.get("f1@5", 0.0),
            "precision_improvement": precision_improvement,
            "recall_improvement": recall_improvement,
            "f1_improvement": f1_improvement,
        }
    except Exception as e:
        logger.error(f"Error calculating overall performance: {e}")
        return {
            "avg_precision@1": 0.0,
            "avg_recall@10": 0.0,
            "reranking_improvement": 0.0,
            "retrieval_quality": 0.0,
            "reranking_quality": 0.0,
            "precision_improvement": 0.0,
            "recall_improvement": 0.0,
            "f1_improvement": 0.0,
        }


def create_per_query_analysis(
    tier1_results: Dict[str, Any],
    tier3_results: Dict[str, Any],
    queries: List[str],
    ground_truth_sets: List[set],
) -> List[Dict[str, Any]]:
    """Tạo phân tích chi tiết cho từng câu hỏi"""
    per_query_results = []

    for i, (query, gt_set) in enumerate(zip(queries, ground_truth_sets)):
        try:
            # Tier 1 results
            tier1_aids = (
                tier1_results["predictions"][i]
                if i < len(tier1_results["predictions"])
                else []
            )
            tier1_scores = (
                tier1_results["scores"][i] if i < len(tier1_results["scores"]) else []
            )

            # Tier 3 results
            tier3_results_list = (
                tier3_results["predictions"][i]
                if i < len(tier3_results["predictions"])
                else []
            )
            tier3_aids = [res["aid"] for res in tier3_results_list]
            tier3_scores = [res.get("rerank_score", 0.0) for res in tier3_results_list]

            # Calculate metrics for each tier
            tier1_metrics = calculate_query_metrics(tier1_aids, gt_set)
            tier3_metrics = calculate_query_metrics(tier3_aids, gt_set)

            per_query_results.append(
                {
                    "query_id": i,
                    "query": query,
                    "ground_truth": list(gt_set),
                    "ground_truth_count": len(gt_set),
                    "tier1_results": {
                        "aids": tier1_aids[:10],
                        "scores": tier1_scores[:10],
                        "precision": tier1_metrics["precision"],
                        "recall": tier1_metrics["recall"],
                        "f1": tier1_metrics["f1"],
                        "found_relevant": len(set(tier1_aids) & gt_set),
                    },
                    "tier3_results": {
                        "aids": tier3_aids,
                        "scores": tier3_scores,
                        "precision": tier3_metrics["precision"],
                        "recall": tier3_metrics["recall"],
                        "f1": tier3_metrics["f1"],
                        "found_relevant": len(set(tier3_aids) & gt_set),
                    },
                    "improvements": {
                        "tier3_vs_tier1": {
                            "precision_improvement": tier3_metrics["precision"]
                            - tier1_metrics["precision"],
                            "recall_improvement": tier3_metrics["recall"]
                            - tier1_metrics["recall"],
                            "f1_improvement": tier3_metrics["f1"] - tier1_metrics["f1"],
                        }
                    },
                    "success": True,
                }
            )

        except Exception as e:
            logger.error(f"Error analyzing query {i}: {e}")
            per_query_results.append(
                {
                    "query_id": i,
                    "query": query,
                    "ground_truth": list(gt_set),
                    "ground_truth_count": len(gt_set),
                    "success": False,
                    "error": str(e),
                }
            )

    return per_query_results


def calculate_query_metrics(
    predicted_aids: List[str], ground_truth: set
) -> Dict[str, float]:
    """Tính toán metrics cho một câu hỏi"""
    if not predicted_aids:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    relevant_found = len(set(predicted_aids) & ground_truth)
    precision = relevant_found / len(predicted_aids) if predicted_aids else 0.0
    recall = relevant_found / len(ground_truth) if ground_truth else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def save_enhanced_report(report: Dict[str, Any]):
    """Lưu báo cáo enhanced"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = (
            config.REPORTS_DIR / f"enhanced_evaluation_report_{timestamp}.json"
        )

        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"[REPORT] Enhanced evaluation report saved to: {report_path}")
        return report_path

    except Exception as e:
        logger.error(f"[REPORT] Error saving enhanced report: {e}")
        return None


def display_enhanced_summary(report: Dict[str, Any]):
    """Hiển thị tóm tắt enhanced evaluation"""
    logger.info("=" * 80)
    logger.info("📊 ENHANCED EVALUATION RESULTS:")
    logger.info("=" * 80)

    # Tier 1 results
    tier1_metrics = report["summary"].get("tier1_metrics", {})
    logger.info("🎯 TIER 1 - Bi-Encoder Retrieval:")
    for k in [1, 3, 5, 10, 20, 50]:
        p = tier1_metrics.get(f"precision@{k}", 0.0)
        r = tier1_metrics.get(f"recall@{k}", 0.0)
        f1 = tier1_metrics.get(f"f1@{k}", 0.0)
        logger.info(f"  Top-{k}: Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}")

    # Tier 3 results
    tier3_metrics = report["summary"].get("tier3_metrics", {})
    logger.info("🎯 TIER 3 - Cross-Encoder Reranking:")
    for k in [1, 3, 5, 10, 20, 50]:
        p = tier3_metrics.get(f"precision@{k}", 0.0)
        r = tier3_metrics.get(f"recall@{k}", 0.0)
        f1 = tier3_metrics.get(f"f1@{k}", 0.0)
        logger.info(f"  Top-{k}: Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}")

    # Overall performance
    overall = report["summary"]["overall_performance"]
    logger.info("📈 OVERALL PERFORMANCE:")
    logger.info(f"  Average Precision@1: {overall.get('avg_precision@1', 0):.4f}")
    logger.info(f"  Average Recall@10: {overall.get('avg_recall@10', 0):.4f}")
    logger.info(
        f"  Tier 3 vs Tier 1 F1 Improvement: {overall.get('f1_improvement', 0):.4f}"
    )

    # Tier analysis
    tier_analysis = report["tier_analysis"]
    logger.info("🔍 TIER ANALYSIS:")
    logger.info(f"  Tier 1 Success: {tier_analysis.get('tier1_success', False)}")
    logger.info(f"  Tier 3 Success: {tier_analysis.get('tier3_success', False)}")

    logger.info("=" * 80)


def main():
    """Hàm chính"""
    logger.info("[START] Starting Enhanced Model Evaluation...")
    # Secondary parse to show chosen mode (optional)
    mode = os.getenv("LAWBOT_PERFORMANCE_MODE", "quality")
    logger.info(f"[CONFIG] PERFORMANCE_MODE={mode}")

    success = run_enhanced_evaluation()

    if success:
        logger.info("✅ Enhanced evaluation completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Enhanced evaluation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
