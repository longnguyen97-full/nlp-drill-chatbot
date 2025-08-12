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
from typing import Dict, List, Any, Optional, Tuple
import random
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
from core.utils.aid_utils import canonicalize_aid_list, canonicalize_aid_set
from core.evaluation_reporter import EvaluationReporter, BatchEvaluator

# Setup logging
logger = get_logger(__name__)


def run_enhanced_evaluation():
    """Chạy đánh giá toàn diện với các cải thiện"""
    logger.info("=" * 80)
    logger.info("ENHANCED MODEL EVALUATION v2.0")
    logger.info("=" * 80)

    try:
        # Parse optional CLI options for faster evaluation and data source
        opts = _parse_eval_cli_options()

        # 1. Khởi tạo pipeline với enhanced error handling
        logger.info("[EVAL] Initializing enhanced pipeline...")
        pipeline = LegalQAPipeline(use_ensemble=True, use_cascaded_reranking=True)
        if not pipeline.is_ready:
            logger.error("[EVAL] Pipeline is not ready. Cannot run evaluation.")
            return False

        # 2. Load evaluation data with fallbacks and sampling
        logger.info("[EVAL] Loading evaluation data...")
        queries, ground_truth_sets, data_path = _load_eval_data_with_fallbacks(
            prefer=opts.get("prefer_source", "validation"),
            max_queries=opts.get("max_queries"),
            sample_seed=opts.get("sample_seed"),
            allow_fallbacks=opts.get("allow_fallbacks", False),
        )
        if not queries:
            logger.error("[EVAL] No evaluation queries available.")
            return False

        # Sanity check: Coverage of ground-truth AIDs in current index
        try:
            index_to_aid_path = config.INDEX_TO_AID_PATH
            if index_to_aid_path.exists():
                with open(index_to_aid_path, "r", encoding="utf-8") as f:
                    from core.utils.aid_utils import canonicalize_aid_ascii
                    index_aids = set(
                        canonicalize_aid_ascii(a) for a in json.load(f)
                    )
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
        if opts.get("fast_eval", True):
            tier1_results = evaluate_tier1_retrieval_batch(
                pipeline,
                queries,
                ground_truth_sets,
                top_k=opts.get("top_k_retrieval") or config.TOP_K_RETRIEVAL,
            )
        else:
            tier1_results = evaluate_tier1_retrieval(pipeline, queries, ground_truth_sets)

        # Tier 2: Light Reranker (only)
        logger.info("[EVAL] === TIER 2: Light Reranker (only) ===")
        tier2_light_results = evaluate_tier2_light_reranking(
            pipeline, queries, ground_truth_sets
        )

        # Tier 3 (Strong Only): Cross-Encoder Reranking WITHOUT Light Reranker
        logger.info(
            "[EVAL] === TIER 3 (Strong Only): Cross-Encoder Reranking (no Light Reranker) ==="
        )
        if opts.get("fast_eval", True):
            tier3_strong_only_results = evaluate_tier3_reranking_batch(
                pipeline,
                queries,
                ground_truth_sets,
                top_k_retrieval=opts.get("top_k_retrieval") or config.TOP_K_RETRIEVAL,
                top_k_final=min(10, opts.get("top_k_final") or 10),
            )
        else:
            tier3_strong_only_results = evaluate_tier3_reranking_strong_only(
                pipeline, queries, ground_truth_sets
            )

        # Cascaded (Tier 2 + Tier 3): Light Reranker + Cross-Encoder Reranking
        logger.info(
            "[EVAL] === CASCADED (Tier 2 + Tier 3): Light Reranker + Cross-Encoder ==="
        )
        if opts.get("fast_eval", True):
            cascaded_results = evaluate_cascaded_reranking_batch(
                pipeline,
                queries,
                ground_truth_sets,
                top_k_retrieval=opts.get("top_k_retrieval") or config.TOP_K_RETRIEVAL,
                top_k_light=opts.get("top_k_light") or config.TOP_K_LIGHT_RERANKING,
                top_k_final=min(10, opts.get("top_k_final") or 10),
            )
        else:
            cascaded_results = evaluate_cascaded_reranking(
                pipeline, queries, ground_truth_sets
            )

        # 4. Xây per-query results phù hợp reporter
        logger.info("[EVAL] Building per-query detailed results...")
        per_query_results = []
        for i, (query, gt_set) in enumerate(zip(queries, ground_truth_sets)):
            try:
                ret_aids = (
                    canonicalize_aid_list(tier1_results["predictions"][i])
                    if i < len(tier1_results["predictions"])
                    else []
                )
                light_only_aids = (
                    tier2_light_results["predictions"][i]
                    if i < len(tier2_light_results["predictions"]) \
                    else []
                )
                strong_only_results = (
                    tier3_strong_only_results["predictions"][i]
                    if i < len(tier3_strong_only_results["predictions"])
                    else []
                )
                cascaded_results_i = (
                    cascaded_results["predictions"][i]
                    if i < len(cascaded_results["predictions"])
                    else []
                )
                # Include retrieval scores captured in Tier 1
                ret_scores = (
                    tier1_results["scores"][i]
                    if i < len(tier1_results.get("scores", []))
                    else []
                )
                # Light-only has no explicit scores; use placeholders
                light_only_scores = []
                strong_only_scores = [res.get("rerank_score", 0.0) for res in strong_only_results]
                cascaded_scores = [res.get("rerank_score", 0.0) for res in cascaded_results_i]

                # Per-query quick metrics
                ret_precision = (len(set(ret_aids) & gt_set) / len(ret_aids)) if ret_aids else 0.0
                ret_recall = (len(set(ret_aids) & gt_set) / len(gt_set)) if gt_set else 0.0
                ret_f1 = (2 * ret_precision * ret_recall / (ret_precision + ret_recall)) if (ret_precision + ret_recall) > 0 else 0.0

                strong_only_aids = canonicalize_aid_list([res.get("aid") for res in strong_only_results])
                strong_only_precision = (len(set(strong_only_aids) & gt_set) / len(strong_only_aids)) if strong_only_aids else 0.0
                strong_only_recall = (len(set(strong_only_aids) & gt_set) / len(gt_set)) if gt_set else 0.0
                strong_only_f1 = (2 * strong_only_precision * strong_only_recall / (strong_only_precision + strong_only_recall)) if (strong_only_precision + strong_only_recall) > 0 else 0.0

                light_only_aids_norm = canonicalize_aid_list(light_only_aids)
                light_only_precision = (len(set(light_only_aids_norm) & gt_set) / len(light_only_aids_norm)) if light_only_aids_norm else 0.0
                light_only_recall = (len(set(light_only_aids_norm) & gt_set) / len(gt_set)) if gt_set else 0.0
                light_only_f1 = (2 * light_only_precision * light_only_recall / (light_only_precision + light_only_recall)) if (light_only_precision + light_only_recall) > 0 else 0.0

                cascaded_aids = canonicalize_aid_list([res.get("aid") for res in cascaded_results_i])
                cascaded_precision = (len(set(cascaded_aids) & gt_set) / len(cascaded_aids)) if cascaded_aids else 0.0
                cascaded_recall = (len(set(cascaded_aids) & gt_set) / len(gt_set)) if gt_set else 0.0
                cascaded_f1 = (2 * cascaded_precision * cascaded_recall / (cascaded_precision + cascaded_recall)) if (cascaded_precision + cascaded_recall) > 0 else 0.0

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
                        "tier3_strong_only_results": {
                            "aids": strong_only_aids,
                            "scores": strong_only_scores,
                            "precision": strong_only_precision,
                            "recall": strong_only_recall,
                            "f1": strong_only_f1,
                            "found_relevant": len(set(strong_only_aids) & gt_set),
                        },
                        "tier2_light_results": {
                            "aids": light_only_aids_norm,
                            "scores": light_only_scores,
                            "precision": light_only_precision,
                            "recall": light_only_recall,
                            "f1": light_only_f1,
                            "found_relevant": len(set(light_only_aids_norm) & gt_set),
                        },
                        "cascaded_results": {
                            "aids": cascaded_aids,
                            "scores": cascaded_scores,
                            "precision": cascaded_precision,
                            "recall": cascaded_recall,
                            "f1": cascaded_f1,
                            "found_relevant": len(set(cascaded_aids) & gt_set),
                        },
                        "improvement": {
                            "tier2_over_tier1": {
                                "precision": light_only_precision - ret_precision,
                                "recall": light_only_recall - ret_recall,
                                "f1": light_only_f1 - ret_f1,
                            },
                            "tier3_over_tier1": {
                                "precision": strong_only_precision - ret_precision,
                                "recall": strong_only_recall - ret_recall,
                                "f1": strong_only_f1 - ret_f1,
                            },
                            "cascaded_over_tier2": {
                                "precision": cascaded_precision - light_only_precision,
                                "recall": cascaded_recall - light_only_recall,
                                "f1": cascaded_f1 - light_only_f1,
                            },
                            "cascaded_over_tier3": {
                                "precision": cascaded_precision - strong_only_precision,
                                "recall": cascaded_recall - strong_only_recall,
                                "f1": cascaded_f1 - strong_only_f1,
                            },
                            "cascaded_over_tier1": {
                                "precision": cascaded_precision - ret_precision,
                                "recall": cascaded_recall - ret_recall,
                                "f1": cascaded_f1 - ret_f1,
                            },
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
            "aid_normalization": "canonical_ascii",
            "data_source": str(data_path) if data_path else "unknown",
            "fast_eval": opts.get("fast_eval", True),
            "top_k_overrides": {
                "retrieval": opts.get("top_k_retrieval"),
                "light": opts.get("top_k_light"),
                "final": opts.get("top_k_final"),
            },
        }
        report = reporter.create_comprehensive_report(
            retrieval_metrics=tier1_results["metrics"],
            reranking_metrics=tier3_strong_only_results["metrics"],
            per_query_results=per_query_results,
            metadata=metadata,
            cascaded_metrics=cascaded_results["metrics"],
            light_metrics=tier2_light_results["metrics"],
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
        # Use configurable evaluation K values from config
        evaluator = BatchEvaluator(k_values=getattr(config, "EVAL_K_VALUES", [1, 3, 5, 10, 20, 50]))
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


def evaluate_tier1_retrieval_batch(
    pipeline: LegalQAPipeline,
    queries: List[str],
    ground_truth_sets: List[set],
    top_k: int,
) -> Dict[str, Any]:
    """Batch retrieval for speed using SentenceTransformer batch encode + FAISS batch.

    Uses `LegalQAPipeline.retrieve_batch` which already optimizes embeddings & FAISS
    calls in a vectorized fashion.
    """
    logger.info("[TIER1] Batch evaluating Bi-Encoder retrieval performance...")

    try:
        retrieved_aids_batch, distances_batch = pipeline.retrieve_batch(queries, top_k)

        evaluator = BatchEvaluator(k_values=getattr(config, "EVAL_K_VALUES", [1, 3, 5, 10, 20, 50]))
        metrics = evaluator.evaluate_batch(queries, ground_truth_sets, retrieved_aids_batch)

        # Convert distances to normalized scores per query
        retrieval_scores = []
        for distances in distances_batch:
            if distances is None or len(distances) == 0:
                retrieval_scores.append([])
                continue
            max_dist = max(distances) if len(distances) > 0 else 1.0
            scores = [1.0 - (d / max_dist) if max_dist > 0 else 0.0 for d in distances]
            retrieval_scores.append(scores)

        return {
            "predictions": retrieved_aids_batch,
            "scores": retrieval_scores,
            "metrics": metrics or {f"precision@{k}": 0.0 for k in [1,3,5,10,20,50]},
            "success": True,
        }
    except Exception as e:
        logger.error(f"[TIER1] Batch retrieval evaluation failed: {e}")
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
        evaluator = BatchEvaluator(k_values=getattr(config, "EVAL_K_VALUES", [1, 3, 5, 10, 20, 50]))
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


def evaluate_tier3_reranking_batch(
    pipeline: LegalQAPipeline,
    queries: List[str],
    ground_truth_sets: List[set],
    top_k_retrieval: int,
    top_k_final: int,
) -> Dict[str, Any]:
    """Faster Tier-3 eval by avoiding light reranker and using vectorized retrieval."""
    logger.info("[TIER3-STRONG] Batch evaluating strong-only reranking...")

    predictions: List[List[Dict[str, Any]]] = []
    scores: List[List[float]] = []

    try:
        # First retrieve for all queries in batch
        retrieved_aids_batch, retrieved_distances_batch = pipeline.retrieve_batch(
            queries, top_k_retrieval
        )

        # Then rerank per query using strong reranker directly
        for q, aids, dists in zip(queries, retrieved_aids_batch, retrieved_distances_batch):
            try:
                results = pipeline.rerank(q, aids, dists)
                results = results[:top_k_final]
                predictions.append(results)
                scores.append([r.get("rerank_score", 0.0) for r in results])
            except Exception as e:
                logger.warning(f"[TIER3-STRONG] Query rerank failed: {e}")
                predictions.append([])
                scores.append([])

        evaluator = BatchEvaluator(k_values=getattr(config, "EVAL_K_VALUES", [1, 3, 5, 10, 20, 50]))
        aids_batch = [[res["aid"] for res in preds] for preds in predictions]
        metrics = evaluator.evaluate_batch(queries, ground_truth_sets, aids_batch)

        return {
            "predictions": predictions,
            "scores": scores,
            "metrics": metrics or {f"precision@{k}": 0.0 for k in [1,3,5,10,20,50]},
            "success": True,
        }
    except Exception as e:
        logger.error(f"[TIER3-STRONG] Batch evaluation failed: {e}")
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

def evaluate_tier2_light_reranking(
    pipeline: LegalQAPipeline, queries: List[str], ground_truth_sets: List[set]
) -> Dict[str, Any]:
    """Đánh giá Tầng 2 - Light Reranker (áp dụng rerank_light lên retrieval)"""
    logger.info("[TIER2] Evaluating Light Reranker performance...")

    predictions = []  # list of light-reranked AIDs per query
    metrics = {}

    try:
        for i, q in enumerate(queries):
            if i % 10 == 0:
                logger.info(f"[TIER2] Processing query {i+1}/{len(queries)}")

            try:
                retrieved_aids, distances = pipeline.retrieve(q, config.TOP_K_RETRIEVAL)
                light_aids, _ = pipeline.rerank_light(
                    q,
                    retrieved_aids,
                    distances,
                    top_k_light=config.TOP_K_LIGHT_RERANKING,
                )
                predictions.append(light_aids)
            except Exception as e:
                logger.error(f"[TIER2] Error in light reranking for query {i}: {e}")
                predictions.append([])

        evaluator = BatchEvaluator(k_values=[1, 3, 5, 10, 20, 50])
        try:
            computed = (
                evaluator.evaluate_batch(queries, ground_truth_sets, predictions)
                if predictions
                else {}
            )
        except Exception as e:
            logger.error(f"[TIER2] Error computing metrics: {e}")
            computed = {}

        if not computed:
            logger.warning("[TIER2] Empty results, using default metrics")
            metrics = {f"precision@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]}
            metrics.update({f"recall@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]})
            metrics.update({f"f1@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]})
        else:
            metrics = computed

        return {
            "predictions": predictions,
            "scores": [],  # light reranker scores are not exposed here
            "metrics": metrics,
            "success": True,
        }

    except Exception as e:
        logger.error(f"[TIER2] Tier 2 evaluation failed: {e}")
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

def evaluate_tier3_reranking_strong_only(
    pipeline: LegalQAPipeline, queries: List[str], ground_truth_sets: List[set]
) -> Dict[str, Any]:
    """Đánh giá Tầng 3 - Strong-only Reranking (KHÔNG dùng Light Reranker)"""
    logger.info("[TIER3-STRONG] Evaluating strong-only reranking performance...")

    predictions = []
    scores = []
    metrics = {}

    try:
        for i, q in enumerate(queries):
            if i % 10 == 0:
                logger.info(f"[TIER3-STRONG] Processing query {i+1}/{len(queries)}")

            try:
                retrieved_aids, distances = pipeline.retrieve(q, config.TOP_K_RETRIEVAL)
                results = pipeline.rerank(q, retrieved_aids, distances)
                predictions.append(results)
                scores.append([res.get("rerank_score", 0.0) for res in results])
            except Exception as e:
                logger.error(f"[TIER3-STRONG] Error in reranking for query {i}: {e}")
                predictions.append([])
                scores.append([])

        evaluator = BatchEvaluator(k_values=[1, 3, 5, 10, 20, 50])
        aids_batch = [[res["aid"] for res in preds] for preds in predictions]

        try:
            computed = (
                evaluator.evaluate_batch(queries, ground_truth_sets, aids_batch)
                if aids_batch
                else {}
            )
        except Exception as e:
            logger.error(f"[TIER3-STRONG] Error computing metrics: {e}")
            computed = {}

        if not computed:
            logger.warning("[TIER3-STRONG] Empty results, using default metrics")
            metrics = {f"precision@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]}
            metrics.update({f"recall@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]})
            metrics.update({f"f1@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]})
        else:
            metrics = computed

        return {
            "predictions": predictions,
            "scores": scores,
            "metrics": metrics,
            "success": True,
        }

    except Exception as e:
        logger.error(f"[TIER3-STRONG] Evaluation failed: {e}")
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


def evaluate_cascaded_reranking(
    pipeline: LegalQAPipeline, queries: List[str], ground_truth_sets: List[set]
) -> Dict[str, Any]:
    """Đánh giá Cascaded: Light Reranker + Strong Reranker"""
    logger.info("[CASCADED] Evaluating cascaded reranking performance...")

    predictions = []
    scores = []
    metrics = {}

    try:
        for i, q in enumerate(queries):
            if i % 10 == 0:
                logger.info(f"[CASCADED] Processing query {i+1}/{len(queries)}")

            try:
                results = pipeline.predict(
                    q,
                    top_k_retrieval=config.TOP_K_RETRIEVAL,
                    top_k_final=10,
                    top_k_light_reranking=config.TOP_K_LIGHT_RERANKING,
                )
                predictions.append(results)
                scores.append([res.get("rerank_score", 0.0) for res in results])
            except Exception as e:
                logger.error(f"[CASCADED] Error in reranking for query {i}: {e}")
                predictions.append([])
                scores.append([])

        evaluator = BatchEvaluator(k_values=[1, 3, 5, 10, 20, 50])
        aids_batch = [[res["aid"] for res in preds] for preds in predictions]

        try:
            computed = (
                evaluator.evaluate_batch(queries, ground_truth_sets, aids_batch)
                if aids_batch
                else {}
            )
        except Exception as e:
            logger.error(f"[CASCADED] Error computing metrics: {e}")
            computed = {}

        if not computed:
            logger.warning("[CASCADED] Empty results, using default metrics")
            metrics = {f"precision@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]}
            metrics.update({f"recall@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]})
            metrics.update({f"f1@{k}": 0.0 for k in [1, 3, 5, 10, 20, 50]})
        else:
            metrics = computed

        return {
            "predictions": predictions,
            "scores": scores,
            "metrics": metrics,
            "success": True,
        }
    except Exception as e:
        logger.error(f"[CASCADED] Evaluation failed: {e}")
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

def evaluate_cascaded_reranking_batch(
    pipeline: LegalQAPipeline,
    queries: List[str],
    ground_truth_sets: List[set],
    top_k_retrieval: int,
    top_k_light: int,
    top_k_final: int,
) -> Dict[str, Any]:
    """Faster cascaded evaluation by batching retrieval first, then light and strong.

    We still call pipeline.predict per query to ensure consistent cascaded logic,
    but retrieval k values can be tuned lower for faster runs via CLI options.
    """
    logger.info("[CASCADED] Batch evaluating cascaded reranking...")

    predictions: List[List[Dict[str, Any]]] = []
    scores: List[List[float]] = []

    try:
        for q in queries:
            try:
                results = pipeline.predict(
                    q,
                    top_k_retrieval=top_k_retrieval,
                    top_k_final=top_k_final,
                    top_k_light_reranking=top_k_light,
                )
                predictions.append(results)
                scores.append([r.get("rerank_score", 0.0) for r in results])
            except Exception as e:
                logger.warning(f"[CASCADED] Query predict failed: {e}")
                predictions.append([])
                scores.append([])

        evaluator = BatchEvaluator(k_values=[1, 3, 5, 10, 20, 50])
        aids_batch = [[res["aid"] for res in preds] for preds in predictions]
        metrics = evaluator.evaluate_batch(queries, ground_truth_sets, aids_batch)

        return {
            "predictions": predictions,
            "scores": scores,
            "metrics": metrics or {f"precision@{k}": 0.0 for k in [1,3,5,10,20,50]},
            "success": True,
        }
    except Exception as e:
        logger.error(f"[CASCADED] Batch evaluation failed: {e}")
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


def _parse_eval_cli_options() -> Dict[str, Any]:
    """Parse simple CLI flags for faster eval and data control without heavy deps.

    Supported flags:
      --fast-eval / --no-fast-eval
      --max-queries N
      --sample-seed N
      --prefer-source validation|train|public_test (khuyến nghị: validation)
      --allow-fallbacks (cho phép fallback sang train/public_test nếu validation không có)
      --topk-retrieval N
      --topk-light N
      --topk-final N
    """
    args = sys.argv[1:]
    opts: Dict[str, Any] = {
        "fast_eval": True,
        "max_queries": None,
        "sample_seed": 42,
        "prefer_source": "validation",
        "allow_fallbacks": False,
        "top_k_retrieval": None,
        "top_k_light": None,
        "top_k_final": None,
    }
    for i, a in enumerate(args):
        if a == "--fast-eval":
            opts["fast_eval"] = True
        elif a == "--no-fast-eval":
            opts["fast_eval"] = False
        elif a.startswith("--max-queries="):
            try:
                opts["max_queries"] = int(a.split("=", 1)[1])
            except Exception:
                pass
        elif a.startswith("--sample-seed="):
            try:
                opts["sample_seed"] = int(a.split("=", 1)[1])
            except Exception:
                pass
        elif a.startswith("--prefer-source="):
            val = a.split("=", 1)[1].strip().lower()
            if val in {"validation", "train", "public_test"}:
                opts["prefer_source"] = val
        elif a == "--allow-fallbacks":
            opts["allow_fallbacks"] = True
        elif a.startswith("--topk-retrieval="):
            try:
                opts["top_k_retrieval"] = int(a.split("=", 1)[1])
            except Exception:
                pass
        elif a.startswith("--topk-light="):
            try:
                opts["top_k_light"] = int(a.split("=", 1)[1])
            except Exception:
                pass
        elif a.startswith("--topk-final="):
            try:
                opts["top_k_final"] = int(a.split("=", 1)[1])
            except Exception:
                pass
    return opts


def _load_eval_data_with_fallbacks(
    prefer: str = "validation",
    max_queries: Optional[int] = None,
    sample_seed: int = 42,
    allow_fallbacks: bool = False,
) -> Tuple[List[str], List[set], Optional[Path]]:
    """Load evaluation data with fallbacks and optional sampling.

    Order of preference (config paths): validation → train → public_test.
    Each item is expected to have fields {"question", "relevant_aids"}. If the
    file is in an older format, gracefully fall back by inferring from available
    fields (e.g., answer_id) and wrapping as a single-element list.
    """
    def _attempt_load(path: Path) -> Optional[List[Dict[str, Any]]]:
        try:
            if path and path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"[EVAL] Failed to load data from {path}: {e}")
        return None

    # Only use validation by default; allow fallbacks if explicitly permitted
    sources: List[Path] = []
    if prefer == "validation":
        sources = [config.VAL_SPLIT_JSON_PATH]
        if allow_fallbacks:
            sources.extend([config.TRAIN_JSON_PATH, config.PUBLIC_TEST_JSON_PATH])
    elif prefer == "train":
        sources = [config.TRAIN_JSON_PATH]
        if allow_fallbacks:
            sources.extend([config.VAL_SPLIT_JSON_PATH, config.PUBLIC_TEST_JSON_PATH])
    else:
        sources = [config.PUBLIC_TEST_JSON_PATH]
        if allow_fallbacks:
            sources.extend([config.VAL_SPLIT_JSON_PATH, config.TRAIN_JSON_PATH])

    data = None
    chosen_path = None
    for p in sources:
        data = _attempt_load(p)
        if data:
            chosen_path = p
            break

    if not data:
        logger.error("[EVAL] Could not load any evaluation data from configured paths.")
        return [], [], None

    # Normalize records: ensure question + relevant_aids
    normalized: List[Tuple[str, List[str]]] = []
    for item in data:
        q = item.get("question") or item.get("query")
        if not q:
            continue
        rel = item.get("relevant_aids")
        if rel is None:
            # Try older format fallbacks
            ans = item.get("answer_id")
            rel = [ans] if ans else []
        if isinstance(rel, str):
            rel = [rel]
        normalized.append((q, list(rel)))

    if not normalized:
        logger.error(f"[EVAL] No valid records in {chosen_path}")
        return [], [], chosen_path

    # Optional sampling for speed
    if max_queries and len(normalized) > max_queries:
        random.seed(sample_seed)
        normalized = random.sample(normalized, k=max_queries)

    queries = [q for q, _ in normalized]
    ground_truth_sets = [canonicalize_aid_set(rel) for _, rel in normalized]

    logger.info(
        f"[EVAL] Using {len(queries)} queries from {chosen_path.name if chosen_path else 'unknown'} (prefer={prefer}, allow_fallbacks={allow_fallbacks})"
    )
    return queries, ground_truth_sets, chosen_path

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
