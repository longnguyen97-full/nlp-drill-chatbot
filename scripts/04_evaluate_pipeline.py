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
import argparse
import pickle
from core.aid_utils import canonicalize_aid_list, canonicalize_aid_set
from core.services.evaluation_service import EvaluationReporter, BatchEvaluator

# Add project root to path
# This needs to run before config is imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Setup logging
# Note: config is now imported in main after arg parsing
from core.services.logging_service import get_logger
from core.retrieval import Retriever
from core.reranking import Reranker

logger = get_logger(__name__)

def _load_aid_map(config) -> Optional[Dict[str, str]]:
    """Loads the AID map from the configured path."""
    if config.AID_MAP_PATH.exists():
        with open(config.AID_MAP_PATH, "rb") as f:
            return pickle.load(f)
    logger.error(f"AID map not found at {config.AID_MAP_PATH}")
    return None

def _run_evaluation_tier(
    tier_name: str,
    eval_func,
    retriever: Retriever,
    reranker: Reranker,
    queries: List[str],
    ground_truth_sets: List[set],
    config,
    aid_map: Dict,
    **kwargs,
) -> Dict[str, Any]:
    """Helper function to run an evaluation tier and handle errors."""
    logger.info(f"[EVAL] === TIER: {tier_name} ===")
    try:
        results = eval_func(
            retriever, reranker, queries, ground_truth_sets, config, aid_map, **kwargs
        )
        if not results.get("success", False):
            logger.error(f"[EVAL] Tier {tier_name} failed. See logs for details.")
        return results
    except Exception as e:
        logger.error(f"[EVAL] Unhandled exception in tier {tier_name}: {e}")
        logger.error(traceback.format_exc())
        return {
            "predictions": [],
            "scores": [],
            "metrics": {},
            "success": False,
            "error": str(e),
        }

def _build_per_query_results(
    queries: List[str],
    ground_truth_sets: List[set],
    tier1_results: Dict[str, Any],
    tier2_light_results: Dict[str, Any],
    tier3_strong_only_results: Dict[str, Any],
    cascaded_results: Dict[str, Any],
) -> List[Dict]:
    """Builds the detailed per-query results structure for the report."""
    per_query_results = []
    for i, (query, gt_set) in enumerate(zip(queries, ground_truth_sets)):
        try:
            # Extract results for each tier, with fallbacks for failed queries
            ret_aids = tier1_results["predictions"][i] if i < len(tier1_results.get("predictions", [])) else []
            light_only_aids = tier2_light_results["predictions"][i] if i < len(tier2_light_results.get("predictions", [])) else []
            strong_only_res = tier3_strong_only_results["predictions"][i] if i < len(tier3_strong_only_results.get("predictions", [])) else []
            cascaded_res = cascaded_results["predictions"][i] if i < len(cascaded_results.get("predictions", [])) else []
            
            strong_only_aids = canonicalize_aid_list([res.get("aid") for res in strong_only_res])
            cascaded_aids = canonicalize_aid_list([res.get("aid") for res in cascaded_res])

            # A simple metric calculator to avoid code duplication
            def calc_metrics(preds, gts):
                if not preds or not gts: return {"precision": 0, "recall": 0, "f1": 0}
                p = len(set(preds) & gts) / len(preds)
                r = len(set(preds) & gts) / len(gts)
                f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0
                return {"precision": p, "recall": r, "f1": f1}

            ret_metrics = calc_metrics(ret_aids, gt_set)
            light_metrics = calc_metrics(light_only_aids, gt_set)
            strong_metrics = calc_metrics(strong_only_aids, gt_set)
            cascaded_metrics = calc_metrics(cascaded_aids, gt_set)

            per_query_results.append({
                "query_id": i, "query": query, "ground_truth": list(gt_set),
                "retrieval_results": {"aids": ret_aids[:10], **ret_metrics},
                "tier2_light_results": {"aids": light_only_aids, **light_metrics},
                "tier3_strong_only_results": {"aids": strong_only_aids, **strong_metrics},
                "cascaded_results": {"aids": cascaded_aids, **cascaded_metrics},
                "success": True
            })
        except Exception as e:
            logger.error(f"[EVAL] Error building per-query result for query {i}: {e}")
            per_query_results.append({"query_id": i, "query": query, "success": False, "error": str(e)})
            
    return per_query_results


def run_enhanced_evaluation(args: argparse.Namespace, config):
    """Chạy đánh giá toàn diện với các cải thiện"""
    logger.info("=" * 80)
    logger.info("ENHANCED MODEL EVALUATION v2.0")
    logger.info("=" * 80)

    try:
        # 1. Load components and aid_map
        retriever = Retriever()
        reranker = Reranker(use_ensemble=True, use_cascaded_reranking=True)
        aid_map = _load_aid_map(config)
        if not aid_map:
            logger.error("[EVAL] Cannot proceed without aid_map.")
            return False

        # 2. Load data
        queries, ground_truth_sets, eval_file_path = _load_eval_data_with_fallbacks(
            config,
            prefer=args.prefer_source,
            max_queries=args.max_queries,
            sample_seed=args.sample_seed,
            allow_fallbacks=args.allow_fallbacks,
        )
        if not queries:
            logger.error("[EVAL] No evaluation queries available.")
            return False
        
        logger.info(f"Loaded {len(queries)} queries for evaluation from {eval_file_path.name}")

        # 3. Run evaluations for all tiers
        tier1_results = _run_evaluation_tier(
            "Bi-Encoder Retrieval",
            evaluate_tier1_retrieval,
            retriever, reranker, queries, ground_truth_sets, config,
            aid_map=aid_map, # Pass aid_map
            top_k=args.topk_retrieval or config.TOP_K_RETRIEVAL,
            use_batch=args.fast_eval,
        )

        tier2_light_results = _run_evaluation_tier(
            "Light Reranker (only)",
            evaluate_tier2_light_reranking,
            retriever, reranker, queries, ground_truth_sets, config,
            aid_map=aid_map  # Pass aid_map
        )

        tier3_strong_only_results = _run_evaluation_tier(
            "Strong Reranker (only)",
            evaluate_tier3_reranking_strong_only,
            retriever, reranker, queries, ground_truth_sets, config,
            aid_map=aid_map, # Pass aid_map
            top_k_retrieval=args.topk_retrieval or config.TOP_K_RETRIEVAL,
            top_k_final=min(10, args.topk_final or 10),
            use_batch=args.fast_eval,
        )

        cascaded_results = _run_evaluation_tier(
            "Cascaded (Light + Strong)",
            evaluate_cascaded_reranking,
            retriever, reranker, queries, ground_truth_sets, config,
            aid_map=aid_map, # Pass aid_map
            top_k_retrieval=args.topk_retrieval or config.TOP_K_RETRIEVAL,
            top_k_light=args.topk_light or config.TOP_K_LIGHT_RERANKING,
            top_k_final=min(10, args.topk_final or 10),
            use_batch=args.fast_eval,
        )

        # 4. Build report
        logger.info("[EVAL] Building final report...")
        per_query_results = _build_per_query_results(
            queries, ground_truth_sets,
            tier1_results, tier2_light_results, tier3_strong_only_results, cascaded_results
        )
        
        reporter = EvaluationReporter()
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "enhanced_comprehensive", "total_queries": len(queries),
            "data_source": str(eval_file_path) if eval_file_path else "unknown",
            "fast_eval": args.fast_eval,
            "top_k_overrides": {
                "retrieval": args.topk_retrieval, "light": args.topk_light, "final": args.topk_final
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
    retriever: Retriever,
    reranker: Reranker, # Not used, but kept for consistent signature
    queries: List[str],
    ground_truth_sets: List[set],
    config,
    aid_map: Dict,
    top_k: int,
    use_batch: bool = True,
) -> Dict[str, Any]:
    """Đánh giá Tầng 1 - Bi-Encoder Retrieval (hỗ trợ cả batch và single mode)."""
    logger.info(f"[TIER1] Evaluating Bi-Encoder retrieval (mode={'batch' if use_batch else 'single'})...")

    retrieved_aids_batch: List[List[str]] = []
    retrieval_scores: List[List[float]] = []

    try:
        if use_batch:
            retrieved_aids_batch, retrieval_scores = retriever.retrieve_batch(queries, top_k)
        else:
            for i, q in enumerate(queries):
                if i % 10 == 0:
                    logger.info(f"[TIER1] Processing query {i+1}/{len(queries)}")
                try:
                    retrieved_aids, scores = retriever.retrieve(q, top_k)
                    retrieved_aids_batch.append(retrieved_aids)
                    retrieval_scores.append(scores)
                except Exception as e:
                    logger.error(f"[TIER1] Error in retrieval for query {i}: {e}")
                    retrieved_aids_batch.append([])
                    retrieval_scores.append([])

        # Calculate metrics
        evaluator = BatchEvaluator(k_values=getattr(config, "EVAL_K_VALUES", [1, 3, 5, 10, 20, 50]))
        metrics = evaluator.evaluate_batch(queries, ground_truth_sets, retrieved_aids_batch)

        return {
            "predictions": retrieved_aids_batch,
            "scores": retrieval_scores,
            "metrics": metrics or {f"precision@{k}": 0.0 for k in [1,3,5,10,20,50]},
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


def evaluate_tier3_reranking_strong_only(
    retriever: Retriever,
    reranker: Reranker,
    queries: List[str],
    ground_truth_sets: List[set],
    config,
    aid_map: Dict,
    top_k_retrieval: int,
    top_k_final: int,
    use_batch: bool = True,
) -> Dict[str, Any]:
    """Đánh giá Tầng 3 - Strong-only Reranking (KHÔNG dùng Light Reranker)."""
    logger.info(f"[TIER3-STRONG] Evaluating strong-only reranking (mode={'batch' if use_batch else 'single'})...")

    predictions: List[List[Dict[str, Any]]] = []
    scores: List[List[float]] = []

    try:
        retrieved_aids_batch, retrieved_scores_batch = (
            retriever.retrieve_batch(queries, top_k_retrieval)
            if use_batch
            else ([], [])
        )

        for i, q in enumerate(queries):
            try:
                aids_to_rerank, scores_to_rerank = (
                    (retrieved_aids_batch[i], retrieved_scores_batch[i])
                    if use_batch
                    else retriever.retrieve(q, top_k_retrieval)
                )

                results = reranker.rerank(q, aids_to_rerank, scores_to_rerank, aid_map)
                results = results[:top_k_final]
                predictions.append(results)
                scores.append([r.get("rerank_score", 0.0) for r in results])
            except Exception as e:
                logger.warning(f"[TIER3-STRONG] Query rerank failed for query {i}: {e}")
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
    retriever: Retriever,
    reranker: Reranker,
    queries: List[str],
    ground_truth_sets: List[set],
    config,
    aid_map: Dict,
    top_k_retrieval: int,
    top_k_light: int,
    top_k_final: int,
    use_batch: bool = True,
) -> Dict[str, Any]:
    """Đánh giá Cascaded: Light Reranker + Strong Reranker."""
    logger.info(f"[CASCADED] Evaluating cascaded reranking (mode={'batch' if use_batch else 'single'})...")

    predictions: List[List[Dict[str, Any]]] = []
    scores: List[List[float]] = []

    try:
        # Batch mode still runs predict query-by-query but can use tuned K values
        for i, q in enumerate(queries):
            if not use_batch and i % 10 == 0:
                logger.info(f"[CASCADED] Processing query {i+1}/{len(queries)}")
            try:
                # Manual pipeline prediction using components
                retrieved_aids, retrieved_scores = retriever.retrieve(q, top_k_retrieval)
                
                if reranker.use_cascaded_reranking:
                    light_aids, light_scores = reranker.rerank_light(
                        q, retrieved_aids, retrieved_scores, aid_map,
                        top_k_light=top_k_light or config.TOP_K_LIGHT_RERANKING
                    )
                    retrieved_aids, retrieved_scores = light_aids, light_scores

                results = reranker.rerank(q, retrieved_aids, retrieved_scores, aid_map)
                results = results[:top_k_final]
                predictions.append(results)
                scores.append([r.get("rerank_score", 0.0) for r in results])
            except Exception as e:
                logger.warning(f"[CASCADED] Query predict failed for query {i}: {e}")
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

def evaluate_tier2_light_reranking(
    retriever: Retriever,
    reranker: Reranker,
    queries: List[str], ground_truth_sets: List[set], config, aid_map: Dict
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
                retrieved_aids, scores = retriever.retrieve(q, config.TOP_K_RETRIEVAL)
                light_aids, _ = reranker.rerank_light(
                    q,
                    retrieved_aids,
                    scores,
                    aid_map,
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

def _load_eval_data_with_fallbacks(
    config,
    prefer: str = "validation",
    max_queries: Optional[int] = None,
    sample_seed: int = 42,
    allow_fallbacks: bool = False,
) -> Tuple[List[str], List[set], Optional[Path]]:
    
    # Tạo int_to_str_id_map từ aid_map.pkl
    aid_map = _load_aid_map(config)
    if not aid_map:
        logger.error("Cannot create int_to_str_id_map because aid_map is missing.")
        return [], [], None
    int_to_str_id_map = {i: s_id for i, s_id in enumerate(aid_map.keys())}

    def _attempt_load(path: Path) -> Optional[List[Dict[str, Any]]]:
        if not path.exists():
            logger.warning(f"Evaluation file not found: {path}")
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not read or parse evaluation file {path}: {e}")
            return None

    def _process_data(data: List[Dict[str, Any]]) -> Tuple[List[str], List[set]]:
        queries = []
        ground_truth_sets = []
        for item in data:
            question = item.get("question")
            relevant_laws_int = item.get("relevant_laws", [])
            if question and relevant_laws_int:
                queries.append(question)
                # Chuyển đổi ground truth từ int ID sang string ID
                gt_aids_str = {int_to_str_id_map.get(int_id) for int_id in relevant_laws_int}
                ground_truth_sets.append({aid for aid in gt_aids_str if aid}) # Loại bỏ các giá trị None
        return queries, ground_truth_sets

    potential_paths = []
    if prefer == "validation":
        potential_paths.extend([config.VAL_SPLIT_JSON_PATH, config.PUBLIC_TEST_JSON_PATH, config.TRAIN_SPLIT_JSON_PATH])
    elif prefer == "train":
        potential_paths.extend([config.TRAIN_JSON_PATH, config.VAL_SPLIT_JSON_PATH, config.PUBLIC_TEST_JSON_PATH])
    else:
        potential_paths.extend([config.PUBLIC_TEST_JSON_PATH, config.VAL_SPLIT_JSON_PATH, config.TRAIN_JSON_PATH])

    data = None
    chosen_path = None
    for p in potential_paths:
        data = _attempt_load(p)
        if data:
            chosen_path = p
            break

    if not data:
        logger.error("[EVAL] Could not load any evaluation data from configured paths.")
        return [], [], None

    queries, ground_truth_sets = _process_data(data)

    if not queries:
        logger.error(f"[EVAL] No valid records in {chosen_path}")
        return [], [], chosen_path

    # Optional sampling for speed
    if max_queries and len(queries) > max_queries:
        random.seed(sample_seed)
        queries, ground_truth_sets = random.sample(list(zip(queries, ground_truth_sets)), k=max_queries)

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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced Model Evaluation Script v2.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "quality"],
        default=os.getenv("LAWBOT_PERFORMANCE_MODE", "quality"),
        help="Performance mode to use. Overrides LAWBOT_PERFORMANCE_MODE env var.",
    )
    parser.add_argument(
        "--fast-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use batch processing for faster evaluation.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of queries to use for evaluation.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for sampling queries.",
    )
    parser.add_argument(
        "--prefer-source",
        choices=["validation", "train", "public_test"],
        default="validation",
        help="Preferred data source for evaluation.",
    )
    parser.add_argument(
        "--allow-fallbacks",
        action="store_true",
        help="Allow falling back to other data sources if preferred one is not found.",
    )
    parser.add_argument(
        "--topk-retrieval", type=int, help="Override Top-K for retrieval."
    )
    parser.add_argument(
        "--topk-light", type=int, help="Override Top-K for light reranking."
    )
    parser.add_argument(
        "--topk-final", type=int, help="Override Top-K for final results."
    )

    return parser.parse_args()


def main():
    """Hàm chính"""
    args = parse_args()

    # Set performance mode from args to influence config loading
    os.environ["LAWBOT_PERFORMANCE_MODE"] = args.mode
    
    # Now that env var is set, we can import config
    import config

    logger.info("[START] Starting Enhanced Model Evaluation...")
    logger.info(f"[CONFIG] PERFORMANCE_MODE={args.mode}")

    success = run_enhanced_evaluation(args, config)

    if success:
        logger.info("✅ Enhanced evaluation completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Enhanced evaluation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
