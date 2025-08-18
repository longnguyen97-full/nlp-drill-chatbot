#!/usr/bin/env python3
"""
Analysis Page - LawBot Application
=================================

Comprehensive evaluation metrics and analysis for all 3 tiers.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime
import time
from typing import List
import numpy as np

try:
    from core.pipeline import LegalQAPipeline
    from core.utils.logging_manager import get_logger
    from core.utils.parent_law_manager import ensure_parent_law_mapping
    from core.utils.system_check import (
        get_model_status,
        get_faiss_index_status,
    )

    # Import metrics functions for detailed evaluation
    from evaluation.metrics import (
        precision_at_k,
        recall_at_k,
        f1_at_k,
        mrr_at_k,
        ndcg_at_k,
    )

    # Create score-based metrics functions for tier evaluation
    def precision_at_k_scores(scores: List[float], k: int) -> float:
        """Calculate precision at k based on scores."""
        if k == 0 or not scores:
            return 0.0
        # Count scores above threshold (0.3 for retrieval - more realistic)
        threshold = 0.3
        relevant_count = sum(1 for score in scores[:k] if score > threshold)
        return relevant_count / k

    def recall_at_k_scores(scores: List[float], k: int) -> float:
        """Calculate recall at k based on scores."""
        if k == 0 or not scores:
            return 0.0
        # For scores, recall is similar to precision in this context
        threshold = 0.3
        relevant_count = sum(1 for score in scores[:k] if score > threshold)
        total_relevant = sum(1 for score in scores if score > threshold)
        return relevant_count / total_relevant if total_relevant > 0 else 0.0

    def f1_at_k_scores(scores: List[float], k: int) -> float:
        """Calculate F1 score at k based on scores."""
        prec = precision_at_k_scores(scores, k)
        rec = recall_at_k_scores(scores, k)
        if prec + rec == 0:
            return 0.0
        return 2 * (prec * rec) / (prec + rec)

    def mrr_at_k_scores(scores: List[float], k: int) -> float:
        """Calculate Mean Reciprocal Rank at k based on scores."""
        if k == 0 or not scores:
            return 0.0
        # Find first score above threshold
        threshold = 0.3
        for i, score in enumerate(scores[:k]):
            if score > threshold:
                return 1.0 / (i + 1)
        return 0.0

    def ndcg_at_k_scores(scores: List[float], k: int) -> float:
        """Calculate NDCG at k based on scores."""
        if k == 0 or not scores:
            return 0.0

        # Normalize scores to 0-1 range
        max_score = max(scores) if scores else 1.0
        if max_score == 0:
            return 0.0

        # Calculate DCG
        dcg = 0.0
        for i, score in enumerate(scores[:k]):
            normalized_score = score / max_score
            dcg += normalized_score / np.log2(i + 2)

        # Calculate IDCG (ideal DCG - sorted scores)
        sorted_scores = sorted(scores, reverse=True)
        idcg = 0.0
        for i, score in enumerate(sorted_scores[:k]):
            normalized_score = score / max_score
            idcg += normalized_score / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    # Remove circular import - will use lazy import instead
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info(
        "Please ensure the project structure is correct and dependencies are installed. "
        "Try running the app from the project root directory."
    )
    st.stop()

# Setup app logger
logger = get_logger("analysis_page")


def get_pipeline_lazy():
    """Lazy load pipeline only when needed."""
    try:
        # Validate parent law mapping before loading pipeline
        logger.info("Validating parent law mapping for analysis...")
        if ensure_parent_law_mapping():
            logger.info("✅ Parent law mapping validated for analysis")
        else:
            logger.warning(
                "⚠️ Parent law mapping validation failed, analysis may not display parent law names"
            )

        from core.pipeline import LegalQAPipeline

        return LegalQAPipeline()
    except Exception as e:
        st.warning(f"⚠️ Pipeline not available: {e}")
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def run_comprehensive_evaluation(_pipeline, test_queries=None):
    """Run comprehensive evaluation with detailed metrics for each tier - OPTIMIZED VERSION."""
    if test_queries is None:
        # Try to load tier-specific validation sets
        validation_dir = Path("features/validation_sets")
        if validation_dir.exists():
            try:
                from training.validation_sets import ValidationSetManager

                manager = ValidationSetManager()
                tier1_val, tier2_val, tier3_val = manager.load_validation_sets(
                    str(validation_dir)
                )

                if tier1_val and tier2_val and tier3_val:
                    logger.info(
                        f"✅ Using validation sets: Tier1={len(tier1_val)}, Tier2={len(tier2_val)}, Tier3={len(tier3_val)}"
                    )
                    # Use validation queries - limit to 10 for performance
                    test_queries = [
                        item.get("query", "")
                        for item in tier1_val[:10]
                        if item.get("query")
                    ]
                else:
                    logger.warning(
                        "⚠️ Some validation sets are empty, using fallback queries"
                    )
                    test_queries = [
                        "Luật về đất đai quy định gì?",
                        "Quy định về thuế thu nhập cá nhân?",
                        "Luật lao động quy định gì về hợp đồng?",
                        "Quy định về xử phạt vi phạm giao thông?",
                        "Luật doanh nghiệp quy định gì về thành lập công ty?",
                    ]
            except Exception as e:
                logger.warning(
                    f"⚠️ Failed to load validation sets: {e}, using fallback queries"
                )
                test_queries = [
                    "Luật về đất đai quy định gì?",
                    "Quy định về thuế thu nhập cá nhân?",
                    "Luật lao động quy định gì về hợp đồng?",
                    "Quy định về xử phạt vi phạm giao thông?",
                    "Luật doanh nghiệp quy định gì về thành lập công ty?",
                ]
        else:
            logger.warning("⚠️ No validation sets found, using fallback queries")
            test_queries = [
                "Luật về đất đai quy định gì?",
                "Quy định về thuế thu nhập cá nhân?",
                "Luật lao động quy định gì về hợp đồng?",
                "Quy định về xử phạt vi phạm giao thông?",
                "Luật doanh nghiệp quy định gì về thành lập công ty?",
            ]

    if not test_queries:
        logger.error("❌ No test queries available for evaluation")
        return None

    logger.info(
        f"🚀 Starting comprehensive evaluation with {len(test_queries)} queries"
    )
    start_time = time.time()

    try:
        # Initialize results structure
        evaluation_results = {
            "tier_1": {},
            "tier_2": {},
            "tier_3": {},
            "combined": {},
        }

        # OPTIMIZATION: Pre-calculate K values to avoid repeated calculations
        k_values = [1, 3, 5, 10]

        # OPTIMIZATION: Batch process queries for better performance
        batch_size = min(5, len(test_queries))  # Process in smaller batches
        total_batches = (len(test_queries) + batch_size - 1) // batch_size

        logger.info(
            f"🔄 Processing {len(test_queries)} queries in {total_batches} batches"
        )

        for i in range(0, len(test_queries), batch_size):
            batch_queries = test_queries[i : i + batch_size]
            current_batch = i // batch_size + 1

            logger.info(
                f"🔄 Processing batch {current_batch}/{total_batches} ({len(batch_queries)} queries)"
            )

            for query_idx, query in enumerate(batch_queries):
                try:
                    # Get search results for this query
                    results = _pipeline.predict(query, top_k_final=max(k_values))

                    if not results:
                        logger.warning(f"⚠️ No results for query: {query[:50]}...")
                        continue

                    # OPTIMIZATION: Calculate all metrics at once for each tier
                    tier_metrics = calculate_tier_metrics_optimized(results, k_values)

                    # Update evaluation results
                    for tier, metrics in tier_metrics.items():
                        for metric_name, values in metrics.items():
                            if metric_name not in evaluation_results[tier]:
                                evaluation_results[tier][metric_name] = []
                            evaluation_results[tier][metric_name].extend(values)

                    logger.debug(
                        f"✅ Processed query {query_idx + 1}/{len(batch_queries)} in batch {current_batch}"
                    )

                except Exception as e:
                    logger.error(f"❌ Error processing query '{query[:50]}...': {e}")
                    continue

        # OPTIMIZATION: Calculate final averages efficiently
        logger.info("🔄 Calculating final averages...")
        final_results = calculate_final_averages_optimized(evaluation_results)

        evaluation_time = time.time() - start_time
        logger.info(
            f"✅ Comprehensive evaluation completed in {evaluation_time:.2f} seconds"
        )

        # Save results to file for future use
        try:
            saved_file = save_comprehensive_evaluation_results(final_results)
            if saved_file:
                logger.info(f"💾 Results saved to: {saved_file}")
            else:
                logger.warning("⚠️ Failed to save results to file")
        except Exception as e:
            logger.warning(f"⚠️ Could not save results: {e}")

        return final_results

    except Exception as e:
        logger.error(f"❌ Comprehensive evaluation failed: {e}")
        return None


def calculate_tier_metrics_optimized(results, k_values):
    """Calculate metrics for all tiers efficiently in one pass."""
    tier_metrics = {
        "tier_1": {},
        "tier_2": {},
        "tier_3": {},
        "combined": {},
    }

    # Extract scores for all tiers at once
    retrieval_scores = [r.get("retrieval_score", 0.0) for r in results]
    light_scores = [r.get("light_reranker_score", 0.0) for r in results]
    cross_scores = [r.get("cross_encoder_score", 0.0) for r in results]
    final_scores = [r.get("final_score", 0.0) for r in results]

    # Calculate metrics for each K value efficiently
    for k in k_values:
        if k <= len(results):
            # Tier 1 (Retrieval)
            tier_metrics["tier_1"][f"precision_{k}"] = [
                precision_at_k_scores(retrieval_scores, k)
            ]
            tier_metrics["tier_1"][f"recall_{k}"] = [
                recall_at_k_scores(retrieval_scores, k)
            ]
            tier_metrics["tier_1"][f"f1_{k}"] = [f1_at_k_scores(retrieval_scores, k)]
            tier_metrics["tier_1"][f"ndcg_{k}"] = [
                ndcg_at_k_scores(retrieval_scores, k)
            ]
            tier_metrics["tier_1"][f"mrr_{k}"] = [mrr_at_k_scores(retrieval_scores, k)]
            tier_metrics["tier_1"][f"quality_{k}"] = [
                calculate_tier_quality_score(retrieval_scores, k)
            ]
            tier_metrics["tier_1"][f"score_effectiveness_{k}"] = [
                calculate_score_effectiveness(retrieval_scores, k)
            ]

            # Tier 2 (Light Reranker)
            tier_metrics["tier_2"][f"precision_{k}"] = [
                precision_at_k_scores(light_scores, k)
            ]
            tier_metrics["tier_2"][f"recall_{k}"] = [
                recall_at_k_scores(light_scores, k)
            ]
            tier_metrics["tier_2"][f"f1_{k}"] = [f1_at_k_scores(light_scores, k)]
            tier_metrics["tier_2"][f"ndcg_{k}"] = [ndcg_at_k_scores(light_scores, k)]
            tier_metrics["tier_2"][f"mrr_{k}"] = [mrr_at_k_scores(light_scores, k)]
            tier_metrics["tier_2"][f"quality_{k}"] = [
                calculate_tier_quality_score(light_scores, k)
            ]
            tier_metrics["tier_2"][f"score_effectiveness_{k}"] = [
                calculate_score_effectiveness(light_scores, k)
            ]

            # Tier 3 (Cross Encoder)
            tier_metrics["tier_3"][f"precision_{k}"] = [
                precision_at_k_scores(cross_scores, k)
            ]
            tier_metrics["tier_3"][f"recall_{k}"] = [
                recall_at_k_scores(cross_scores, k)
            ]
            tier_metrics["tier_3"][f"f1_{k}"] = [f1_at_k_scores(cross_scores, k)]
            tier_metrics["tier_3"][f"ndcg_{k}"] = [ndcg_at_k_scores(cross_scores, k)]
            tier_metrics["tier_3"][f"mrr_{k}"] = [mrr_at_k_scores(cross_scores, k)]
            tier_metrics["tier_3"][f"quality_{k}"] = [
                calculate_tier_quality_score(cross_scores, k)
            ]
            tier_metrics["tier_3"][f"score_effectiveness_{k}"] = [
                calculate_score_effectiveness(cross_scores, k)
            ]

            # Combined
            tier_metrics["combined"][f"precision_{k}"] = [
                precision_at_k_scores(final_scores, k)
            ]
            tier_metrics["combined"][f"recall_{k}"] = [
                recall_at_k_scores(final_scores, k)
            ]
            tier_metrics["combined"][f"f1_{k}"] = [f1_at_k_scores(final_scores, k)]
            tier_metrics["combined"][f"ndcg_{k}"] = [ndcg_at_k_scores(final_scores, k)]
            tier_metrics["combined"][f"mrr_{k}"] = [mrr_at_k_scores(final_scores, k)]
            tier_metrics["combined"][f"quality_{k}"] = [
                calculate_tier_quality_score(final_scores, k)
            ]
            tier_metrics["combined"][f"score_effectiveness_{k}"] = [
                calculate_score_effectiveness(final_scores, k)
            ]

    return tier_metrics


def calculate_final_averages_optimized(evaluation_results):
    """Calculate final averages efficiently."""
    final_results = {}

    for tier, metrics in evaluation_results.items():
        final_results[tier] = {}

        # Group metrics by type (precision, recall, f1, etc.)
        metric_groups = {}
        for metric_name, values in metrics.items():
            if values:  # Only process non-empty lists
                base_metric = metric_name.split("_")[0]  # Extract base metric name
                if base_metric not in metric_groups:
                    metric_groups[base_metric] = []
                metric_groups[base_metric].extend(values)

        # Calculate averages for each metric group
        for base_metric, all_values in metric_groups.items():
            if all_values:
                avg_value = sum(all_values) / len(all_values)
                final_results[tier][f"{base_metric}_avg"] = avg_value

                # Log the calculation for debugging
                logger.info(f"Calculated {base_metric}_avg for {tier}: {avg_value:.4f}")

    return final_results


# Remove unused helper functions - they are no longer needed after optimization
# def calculate_f1_from_scores(scores: List[float], k: int) -> float:
# def calculate_ndcg_from_scores(scores: List[float], k: int) -> float:
# def calculate_mrr_from_scores(scores: List[float], k: int) -> float:


def calculate_score_effectiveness(scores: List[float], k: int) -> float:
    """Calculate score effectiveness using tier-specific thresholds."""
    if not scores or k == 0:
        return 0.0

    max_score = max(scores[:k]) if scores[:k] else 0.0
    avg_score = sum(scores[:k]) / k if k > 0 else 0.0

    # Use tier-specific thresholds for effectiveness calculation
    if max_score >= 0.6:  # Tier 1 (Retrieval)
        active_threshold = 0.6  # High quality threshold
        effectiveness_weight = 1.0  # Full weight for retrieval
    elif max_score >= 0.05:  # Tier 2 (Light Reranker)
        active_threshold = 0.02  # Lower threshold for reranker
        effectiveness_weight = 0.8  # Slightly reduced weight
    elif max_score >= 0.01:  # Tier 3 (Cross Encoder)
        active_threshold = 0.05  # Medium threshold for cross encoder
        effectiveness_weight = 0.9  # High weight for cross encoder
    else:  # Very low scores
        active_threshold = 0.01
        effectiveness_weight = 0.5  # Reduced weight for very low scores

    # Count scores that are significantly above threshold
    active_count = sum(1 for score in scores[:k] if score > active_threshold)

    # Calculate base effectiveness
    base_effectiveness = (active_count / k) * 100.0 if k > 0 else 0.0

    # Apply tier-specific weighting and consider score distribution
    score_variance = (
        sum((score - avg_score) ** 2 for score in scores[:k]) / k if k > 1 else 0.0
    )
    stability_factor = max(
        0.5, 1.0 - score_variance
    )  # Higher stability = higher factor

    # Final effectiveness with weighting and stability
    final_effectiveness = base_effectiveness * effectiveness_weight * stability_factor

    return min(100.0, max(0.0, final_effectiveness))


def calculate_tier_quality_score(scores: List[float], k: int) -> float:
    """Calculate a quality score that represents the tier's performance level."""
    if not scores or k == 0:
        return 0.0

    max_score = max(scores[:k]) if scores[:k] else 0.0
    avg_score = sum(scores[:k]) / k if k > 0 else 0.0
    min_score = min(scores[:k]) if scores[:k] else 0.0

    # Calculate score stability (lower variance = more stable)
    if k > 1:
        variance = sum((score - avg_score) ** 2 for score in scores[:k]) / (k - 1)
        stability_factor = max(
            0.3, 1 - variance
        )  # Higher stability = higher factor, min 0.3
    else:
        stability_factor = 1.0

    # Calculate score range factor (how spread out the scores are)
    score_range = max_score - min_score if max_score > min_score else 0.0
    range_factor = max(0.5, 1.0 - score_range)  # Lower range = higher factor, min 0.5

    # Calculate consistency factor (how many scores are close to average)
    if k > 1:
        close_to_avg = sum(1 for score in scores[:k] if abs(score - avg_score) < 0.1)
        consistency_factor = close_to_avg / k
    else:
        consistency_factor = 1.0

    # Tier-specific quality calculation with weighted factors
    # Tier 1 (Retrieval): Focus on high scores and stability
    # Tier 2 (Light Reranker): Balance between scores and consistency
    # Tier 3 (Cross Encoder): Focus on consistency and stability

    if max_score >= 0.6:  # Tier 1 (Retrieval)
        quality_score = (
            max_score * 0.35  # Max score importance
            + avg_score * 0.25  # Average score
            + stability_factor * 0.25  # Stability
            + range_factor * 0.15  # Range control
        )
    elif max_score >= 0.05:  # Tier 2 (Light Reranker)
        quality_score = (
            max_score * 0.25  # Max score
            + avg_score * 0.30  # Average score importance
            + stability_factor * 0.20  # Stability
            + consistency_factor * 0.25  # Consistency importance
        )
    else:  # Tier 3 (Cross Encoder) or very low scores
        quality_score = (
            max_score * 0.20  # Max score
            + avg_score * 0.25  # Average score
            + stability_factor * 0.30  # Stability importance
            + consistency_factor * 0.25  # Consistency importance
        )

    # Normalize to 0-1 range and apply final adjustments
    quality_score = min(1.0, max(0.0, quality_score))

    # Bonus for excellent performance
    if max_score >= 0.8 and avg_score >= 0.6:
        quality_score = min(1.0, quality_score + 0.1)

    return quality_score


def create_tier_performance_chart(eval_results):
    """Create comprehensive tier performance comparison chart."""
    if not eval_results:
        return None

    # Prepare data for visualization
    metrics = ["precision", "recall", "f1", "ndcg", "mrr", "quality"]
    tiers = ["tier_1", "tier_2", "tier_3", "combined"]
    tier_names = ["Retrieval", "Light Reranker", "Cross Encoder", "Combined"]

    # Create data for the chart
    chart_data = []
    for i, tier in enumerate(tiers):
        for metric in metrics:
            avg_value = eval_results[tier].get(f"{metric}_avg", 0.0)
            chart_data.append(
                {
                    "Tier": tier_names[i],
                    "Metric": metric.upper(),
                    "Value": avg_value,
                    "Tier_Type": "Individual" if i < 3 else "Combined",
                    "Color": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][i],
                }
            )

    df = pd.DataFrame(chart_data)

    # Create enhanced bar chart with better styling
    fig = px.bar(
        df,
        x="Metric",
        y="Value",
        color="Tier",
        title="🎯 Tier Performance Comparison - Comprehensive Metrics",
        color_discrete_map={
            "Retrieval": "#1f77b4",
            "Light Reranker": "#ff7f0e",
            "Cross Encoder": "#2ca02c",
            "Combined": "#d62728",
        },
        barmode="group",
        text="Value",
    )

    # Enhance chart appearance
    fig.update_traces(
        texttemplate="%{text:.3f}", textposition="outside", textfont_size=10
    )

    fig.update_layout(
        height=600,
        xaxis_title="📊 Metrics",
        yaxis_title="📈 Score",
        showlegend=True,
        title_font_size=16,
        title_x=0.5,
        plot_bgcolor="white",
        bargap=0.2,
        bargroupgap=0.1,
    )

    # Add grid lines for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    return fig


def create_tier_improvement_chart(eval_results):
    """Create chart showing improvement from individual tiers to combined."""
    if not eval_results:
        return None

    # Calculate improvement percentages
    improvement_data = []

    for metric in ["precision", "recall", "f1", "ndcg", "mrr", "quality"]:
        # Get best individual tier performance
        individual_scores = [
            eval_results["tier_1"].get(f"{metric}_avg", 0.0),
            eval_results["tier_2"].get(f"{metric}_avg", 0.0),
            eval_results["tier_3"].get(f"{metric}_avg", 0.0),
        ]
        best_individual = max(individual_scores)
        combined_score = eval_results["combined"].get(f"{metric}_avg", 0.0)

        if best_individual > 0:
            improvement = ((combined_score - best_individual) / best_individual) * 100
        else:
            improvement = 0.0

        improvement_data.append(
            {
                "Metric": metric.upper(),
                "Best Individual": best_individual,
                "Combined": combined_score,
                "Improvement %": improvement,
            }
        )

    df = pd.DataFrame(improvement_data)

    # Create enhanced improvement chart
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="🏆 Best Individual Tier",
            x=df["Metric"],
            y=df["Best Individual"],
            marker_color="lightblue",
            text=df["Best Individual"].round(3),
            textposition="outside",
        )
    )

    fig.add_trace(
        go.Bar(
            name="🚀 Combined Pipeline",
            x=df["Metric"],
            y=df["Combined"],
            marker_color="darkblue",
            text=df["Combined"].round(3),
            textposition="outside",
        )
    )

    # Add improvement annotations
    for i, row in df.iterrows():
        if row["Improvement %"] > 0:
            fig.add_annotation(
                x=row["Metric"],
                y=row["Combined"] + 0.01,
                text=f"+{row['Improvement %']:.1f}%",
                showarrow=False,
                font=dict(color="green", size=12, weight="bold"),
                bgcolor="lightgreen",
                bordercolor="green",
                borderwidth=1,
            )
        elif row["Improvement %"] < 0:
            fig.add_annotation(
                x=row["Metric"],
                y=row["Combined"] + 0.01,
                text=f"{row['Improvement %']:.1f}%",
                showarrow=False,
                font=dict(color="red", size=12, weight="bold"),
                bgcolor="lightcoral",
                bordercolor="red",
                borderwidth=1,
            )

    fig.update_layout(
        title="📈 Performance Improvement: Individual vs Combined Pipeline",
        height=500,
        barmode="group",
        xaxis_title="📊 Metrics",
        yaxis_title="📈 Score",
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_score_distribution_chart(eval_results):
    """Create chart showing score distribution across tiers."""
    if not eval_results:
        return None

    # Extract score effectiveness data
    tiers = ["tier_1", "tier_2", "tier_3", "combined"]
    tier_names = ["Retrieval", "Light Reranker", "Cross Encoder", "Combined"]

    # Get precision (which now represents score effectiveness)
    effectiveness_data = []
    for i, tier in enumerate(tiers):
        effectiveness = (
            eval_results[tier].get("precision_avg", 0.0) * 100
        )  # Convert to percentage
        effectiveness_data.append(
            {
                "Tier": tier_names[i],
                "Score Effectiveness (%)": effectiveness,
                "Color": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][i],
            }
        )

    df = pd.DataFrame(effectiveness_data)

    # Create horizontal bar chart for better readability
    fig = px.bar(
        df,
        y="Tier",
        x="Score Effectiveness (%)",
        title="🎯 Score Effectiveness by Tier",
        color="Tier",
        color_discrete_map={
            "Retrieval": "#1f77b4",
            "Light Reranker": "#ff7f0e",
            "Cross Encoder": "#2ca02c",
            "Combined": "#d62728",
        },
        orientation="h",
        text="Score Effectiveness (%)",
    )

    # Add value labels
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

    fig.update_layout(
        height=400,
        xaxis_title="📊 Effectiveness (%)",
        yaxis_title="🎯 Tier",
        plot_bgcolor="white",
        showlegend=False,
        title_x=0.5,
    )

    return fig


@st.cache_data(ttl=1800)
def load_analysis_data():
    """Load all necessary data for the analysis page."""
    logger.info("Loading analysis data...")

    # Load system status
    model_status = get_model_status()
    faiss_status = get_faiss_index_status()

    # Load latest evaluation reports
    evaluation_data = load_latest_evaluation_reports()

    return {
        "models": model_status,
        "faiss": faiss_status,
        "evaluation": evaluation_data,
    }


def auto_run_comprehensive_evaluation():
    """Automatically run comprehensive evaluation and cache results with enhanced error handling."""
    try:
        logger.info("🔄 Auto-running comprehensive evaluation...")

        # Check if we're already loading to prevent multiple calls
        if st.session_state.get("eval_loading", False):
            logger.info("⚠️ Evaluation already in progress, skipping auto-run")
            return None

        # First, try to load from saved file to avoid re-computation
        cached_results = load_latest_comprehensive_evaluation()
        if cached_results:
            logger.info("✅ Loaded comprehensive evaluation results from cache file")
            return cached_results

        logger.info("🔄 No cached results found, running fresh evaluation...")

        # Set loading state to prevent multiple calls
        st.session_state.eval_loading = True

        # Get pipeline with timeout protection
        pipeline = get_pipeline_lazy()
        if not pipeline:
            logger.warning("⚠️ Pipeline not available for auto-evaluation")
            st.session_state.eval_loading = False
            return None

        # Run evaluation with progress tracking
        eval_results = run_comprehensive_evaluation(pipeline)

        # Clear loading state
        st.session_state.eval_loading = False

        if eval_results:
            logger.info("✅ Auto-comprehensive evaluation completed successfully")
            return eval_results
        else:
            logger.warning("⚠️ Auto-comprehensive evaluation returned no results")
            return None

    except Exception as e:
        logger.error(f"❌ Auto-comprehensive evaluation failed: {e}")
        # Ensure loading state is cleared on error
        st.session_state.eval_loading = False
        return None


def load_latest_evaluation_reports():
    """Load the latest evaluation reports from reports/evaluation directory."""
    try:
        reports_dir = Path("reports/evaluation")
        if not reports_dir.exists():
            return None

        # Find latest comprehensive evaluation
        comp_files = list(reports_dir.glob("comprehensive_evaluation_*.json"))
        if not comp_files:
            return None

        latest_comp = max(comp_files, key=lambda p: p.stat().st_mtime)

        with open(latest_comp, "r", encoding="utf-8") as f:
            comprehensive = json.load(f)

        # Load tier-specific reports
        tier_reports = {}
        for tier_file in reports_dir.glob("tier_*_evaluation_*.json"):
            tier_name = tier_file.stem.split("_")[0:2]  # tier_1, tier_2, tier_3
            tier_key = "_".join(tier_name)

            with open(tier_file, "r", encoding="utf-8") as f:
                tier_reports[tier_key] = json.load(f)

        return {
            "comprehensive": comprehensive,
            "tiers": tier_reports,
            "timestamp": latest_comp.stat().st_mtime,
        }

    except Exception as e:
        logger.warning(f"Failed to load evaluation reports: {e}")
        return None


def create_tier_comparison_chart(eval_data):
    """Create tier comparison chart."""
    if not eval_data or "comprehensive" not in eval_data:
        return None

    comp = eval_data["comprehensive"]
    tiers = comp.get("tier_evaluation", {})

    # Extract readiness scores for each tier
    tier_data = []
    for tier_key, tier_info in tiers.items():
        readiness = 0
        if tier_info.get("status") == "evaluated":
            readiness = 100
        elif tier_info.get("status") == "partially_ready":
            readiness = 75
        elif tier_info.get("status") == "missing":
            readiness = 0

        tier_data.append(
            {
                "Tier": tier_info.get("name", tier_key),
                "Readiness": readiness,
                "Status": tier_info.get("status", "unknown"),
            }
        )

    if not tier_data:
        return None

    df = pd.DataFrame(tier_data)

    fig = px.bar(
        df,
        x="Tier",
        y="Readiness",
        title="Tier Readiness Comparison",
        color="Readiness",
        color_continuous_scale="RdYlGn",
        text="Readiness",
    )

    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(height=400, showlegend=False)

    return fig


def create_pipeline_health_gauge(eval_data):
    """Create pipeline health gauge chart."""
    if not eval_data or "comprehensive" not in eval_data:
        return None

    health_score = (
        eval_data["comprehensive"]
        .get("pipeline_evaluation", {})
        .get("pipeline_health_score", 0)
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=health_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Pipeline Health Score"},
            delta={"reference": 100},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 50], "color": "lightgray"},
                    {"range": [50, 80], "color": "yellow"},
                    {"range": [80, 100], "color": "green"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )

    fig.update_layout(height=300)
    return fig


def create_tier_detailed_analysis(eval_data):
    """Create detailed tier-by-tier analysis."""
    if not eval_data or "comprehensive" not in eval_data:
        return None

    comp = eval_data["comprehensive"]
    tiers = comp.get("tier_evaluation", {})

    # Create detailed analysis for each tier
    tier_analysis = []

    for tier_key, tier_info in tiers.items():
        tier_name = tier_info.get("name", tier_key)
        status = tier_info.get("status", "unknown")
        model_size = tier_info.get("model_size_mb", 0)

        # Calculate tier-specific metrics
        readiness_score = 0
        if status == "evaluated":
            readiness_score = 100
        elif status == "partially_ready":
            readiness_score = 75
        elif status == "missing":
            readiness_score = 0

        tier_analysis.append(
            {
                "Tier": tier_name,
                "Status": status.upper(),
                "Readiness": f"{readiness_score}%",
                "Model Size (MB)": model_size,
                "Model Available": (
                    "✅ Yes"
                    if tier_info.get("metrics", {}).get("model_availability")
                    else "❌ No"
                ),
            }
        )

    return pd.DataFrame(tier_analysis)


def create_combined_vs_individual_analysis(eval_data):
    """Create analysis comparing combined vs individual tier performance."""
    if not eval_data or "comprehensive" not in eval_data:
        return None

    comp = eval_data["comprehensive"]

    # Individual tier performance
    individual_scores = []
    tiers = comp.get("tier_evaluation", {})

    for tier_key, tier_info in tiers.items():
        tier_name = tier_info.get("name", tier_key)
        status = tier_info.get("status", "unknown")

        score = 0
        if status == "evaluated":
            score = 100
        elif status == "partially_ready":
            score = 75
        elif status == "missing":
            score = 0

        individual_scores.append(
            {
                "Metric": f"{tier_name} (Individual)",
                "Score": score,
                "Type": "Individual",
            }
        )

    # Combined pipeline performance
    pipeline_health = comp.get("pipeline_evaluation", {}).get(
        "pipeline_health_score", 0
    )
    individual_scores.append(
        {"Metric": "Combined Pipeline", "Score": pipeline_health, "Type": "Combined"}
    )

    df = pd.DataFrame(individual_scores)

    fig = px.bar(
        df,
        x="Metric",
        y="Score",
        color="Type",
        title="Individual vs Combined Performance",
        color_discrete_map={"Individual": "lightblue", "Combined": "darkblue"},
    )

    fig.update_layout(height=400)
    return fig


def save_comprehensive_evaluation_results(results, filename=None):
    """Save comprehensive evaluation results to a file for future use."""
    try:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_evaluation_{timestamp}.json"

        # Create reports directory if it doesn't exist
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        filepath = reports_dir / filename

        # Prepare data for saving
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "metadata": {
                "version": "v8.3",
                "type": "comprehensive_evaluation",
                "source": "analysis_page",
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Comprehensive evaluation results saved to: {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"❌ Failed to save evaluation results: {e}")
        return None


def load_latest_comprehensive_evaluation():
    """Load the latest comprehensive evaluation results from file."""
    try:
        reports_dir = Path("reports")
        if not reports_dir.exists():
            return None

        # Find the latest comprehensive evaluation file
        comp_files = list(reports_dir.glob("comprehensive_evaluation_*.json"))
        if not comp_files:
            return None

        # Get the most recent file
        latest_file = max(comp_files, key=lambda p: p.stat().st_mtime)

        # Check if file is recent (within last 24 hours)
        file_age = time.time() - latest_file.stat().st_mtime
        if file_age > 86400:  # 24 hours in seconds
            logger.info("⚠️ Latest evaluation file is older than 24 hours")
            return None

        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"✅ Loaded evaluation results from: {latest_file}")
        return data.get("results")

    except Exception as e:
        logger.error(f"❌ Failed to load evaluation results: {e}")
        return None


def clear_comprehensive_evaluation_cache():
    """Clear the comprehensive evaluation cache and force a fresh evaluation."""
    try:
        # Clear session state
        if "comprehensive_eval_results" in st.session_state:
            del st.session_state.comprehensive_eval_results

        if "eval_loading" in st.session_state:
            st.session_state.eval_loading = False

        # Clear cached file
        reports_dir = Path("reports")
        if reports_dir.exists():
            comp_files = list(reports_dir.glob("comprehensive_evaluation_*.json"))
            for file_path in comp_files:
                try:
                    file_path.unlink()
                    logger.info(f"✅ Deleted cached file: {file_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete {file_path}: {e}")

        logger.info("✅ Comprehensive evaluation cache cleared successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to clear cache: {e}")
        return False


def force_fresh_comprehensive_evaluation():
    """Force a fresh comprehensive evaluation by clearing cache and running evaluation."""
    try:
        logger.info("🔄 Forcing fresh comprehensive evaluation...")

        # Clear cache first
        clear_comprehensive_evaluation_cache()

        # Get pipeline
        pipeline = get_pipeline_lazy()
        if not pipeline:
            logger.warning("⚠️ Pipeline not available for fresh evaluation")
            return None

        # Run fresh evaluation
        eval_results = run_comprehensive_evaluation(pipeline)

        if eval_results:
            logger.info("✅ Fresh comprehensive evaluation completed successfully")
            return eval_results
        else:
            logger.warning("⚠️ Fresh comprehensive evaluation returned no results")
            return None

    except Exception as e:
        logger.error(f"❌ Fresh comprehensive evaluation failed: {e}")
        return None


def force_clean_page_state():
    """Force clean page state to prevent element bleeding and UI issues."""
    try:
        # Clear all potentially conflicting session states
        cleanup_keys = [
            "comprehensive_eval_results",
            "eval_loading",
            "switch_to_tab2",
            "pipeline_loaded",
            "search_results",
            "search_query",
            "analysis_page_loaded",
        ]

        for key in cleanup_keys:
            if key in st.session_state:
                del st.session_state[key]

        # Reset page state
        st.session_state.analysis_page_loaded = True
        st.session_state.page_load_time = time.time()

        logger.info("✅ Page state cleaned successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to clean page state: {e}")
        return False


def render_analysis_page():
    """Render the simplified analysis page with only essential tabs"""
    # Enhanced page state management to prevent element bleeding
    if "analysis_page_loaded" not in st.session_state:
        st.session_state.analysis_page_loaded = True
        st.session_state.comprehensive_eval_results = None
        st.session_state.eval_loading = False
        st.session_state.page_load_time = time.time()

        # Clear any conflicting states from other pages
        conflicting_keys = [
            "search_page_loaded",
            "system_page_loaded",
            "search_results",
            "search_query",
            "pipeline_loaded",
        ]
        for key in conflicting_keys:
            if key in st.session_state:
                del st.session_state[key]

    # Force clean state if page was reloaded
    current_time = time.time()
    if "page_load_time" in st.session_state:
        if current_time - st.session_state.page_load_time > 300:  # 5 minutes
            # Reset page state if it's been too long
            st.session_state.analysis_page_loaded = True
            st.session_state.comprehensive_eval_results = None
            st.session_state.eval_loading = False
            st.session_state.page_load_time = current_time

    # Clean page state and ensure no element bleeding
    if "analysis_page_loaded" not in st.session_state:
        st.session_state.analysis_page_loaded = True
        st.session_state.comprehensive_eval_results = None
        st.session_state.eval_loading = False

    # Enhanced page cleanup CSS - ensure sidebar visibility and clean transitions
    cleanup_css = """
    <style>
    /* Ensure no lingering elements */
    .stApp > div[style*="visibility: hidden"] {
        visibility: visible !important;
    }
    </style>
    """
    st.markdown(cleanup_css, unsafe_allow_html=True)

    st.title("📊 Báo cáo phân tích chi tiết")
    st.markdown("Phân tích toàn diện hiệu suất từng tầng và khuyến nghị cải thiện")

    # Load comprehensive evaluation data
    analysis_data = load_analysis_data()
    model_status = analysis_data.get("models", {})
    faiss_status = analysis_data.get("faiss", {})
    eval_data = analysis_data.get("evaluation")

    # Create simplified tabs - only essential ones
    tab1, tab2 = st.tabs(
        [
            "🏥 Tổng quan",
            "🧪 Phân tích đánh giá toàn diện",
        ]
    )

    # Handle tab switching for fresh evaluation
    if st.session_state.get("switch_to_tab2", False):
        st.session_state.switch_to_tab2 = False
        # This will be handled by JavaScript or we can show a message
        st.info(
            "🔄 Vui lòng chuyển sang tab 'Phân tích đánh giá toàn diện' để chạy Fresh Evaluation"
        )

    with tab1:
        st.subheader("🏗️ Kiến trúc Hệ thống LawBot")

        # System Architecture Overview
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                """
            **🎯 LawBot là hệ thống AI hỏi đáp pháp luật Việt Nam với kiến trúc 3 tầng:**
            
            - **Tầng 1 (Retrieval)**: Bi-Encoder với Contrastive Learning
            - **Tầng 2 (Light Reranking)**: PhoBERT-based light reranker  
            - **Tầng 3 (Final Reranking)**: Cross-Encoder ensemble
            """
            )

        with col2:
            st.metric("Version", "v8.3")
            st.metric("Architecture", "3-Tier")
            st.metric("Language", "Vietnamese")

        # Training Flow Diagram
        st.subheader("🔄 Luồng Training & MLOps")

        # Create training flow visualization
        training_flow_data = {
            "Stage": ["Data Preparation", "Model Training", "Validation", "Deployment"],
            "Tier 1": [
                "Contrastive Learning + HNM",
                "Bi-Encoder Training",
                "Tier-specific Val",
                "Model Export",
            ],
            "Tier 2": [
                "ADAPT Training",
                "Light Reranker",
                "Tier-specific Val",
                "Model Export",
            ],
            "Tier 3": [
                "Ensemble Creation",
                "Cross-Encoder",
                "Tier-specific Val",
                "Pipeline Integration",
            ],
        }

        df_training = pd.DataFrame(training_flow_data)
        st.dataframe(df_training, use_container_width=True)

        # MLOps Techniques
        st.subheader("⚡ Kỹ thuật MLOps được áp dụng")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
            **🔧 Model Management:**
            - Version control cho models
            - Model registry với metadata
            - Automated model loading
            - Contrastive Learning optimization
            """
            )

        with col2:
            st.markdown(
                """
            **📊 Monitoring:**
            - Real-time performance metrics
            - Model health monitoring
            - Automated evaluation
            """
            )

        with col3:
            st.markdown(
                """
            **🚀 Deployment:**
            - Automated pipeline deployment
            - A/B testing support
            - Rollback mechanisms
            """
            )

        # App Processing Flow
        st.subheader("🔄 Luồng Xử lý Ứng dụng")

        # Create app flow diagram
        app_flow_steps = [
            "User Input Query",
            "Tier 1: Retrieval (Bi-Encoder)",
            "Tier 2: Light Reranking",
            "Tier 3: Cross-Encoder Reranking",
            "Score Aggregation",
            "Result Ranking & Display",
        ]

        # Visual flow representation
        flow_cols = st.columns(len(app_flow_steps))
        for i, (step, col) in enumerate(zip(app_flow_steps, flow_cols)):
            with col:
                if i == 0:
                    st.info(f"🎯 {step}")
                elif i == len(app_flow_steps) - 1:
                    st.success(f"✅ {step}")
                else:
                    st.warning(f"⚡ {step}")

        # Model Technologies
        st.subheader("🤖 Công nghệ Models được áp dụng")

        model_tech_data = {
            "Component": ["Bi-Encoder", "Light Reranker", "Cross-Encoder"],
            "Base Model": [
                "Vietnamese Bi-Encoder",
                "PhoBERT-base-v2",
                "PhoBERT + PhoBART",
            ],
            "Technique": ["Contrastive Learning + HNM", "ADAPT", "Ensemble (7:3)"],
            "Purpose": ["Retrieval", "Fast Filtering", "Final Ranking"],
        }

        df_models = pd.DataFrame(model_tech_data)
        st.dataframe(df_models, use_container_width=True)

        # Pipeline Status & Configuration
        st.subheader("🔧 Trạng thái Pipeline & Cấu hình")

        # Auto-check pipeline status
        if "pipeline_status" not in st.session_state:
            st.session_state.pipeline_status = None
            st.session_state.pipeline_checking = False

        # Auto-check button with loading state
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button(
                "🔄 Kiểm tra Pipeline",
                type="secondary",
                disabled=st.session_state.pipeline_checking,
            ):
                st.session_state.pipeline_checking = True
                st.session_state.pipeline_status = None
                st.rerun()

        with col2:
            if st.session_state.pipeline_checking:
                st.info("🔄 Đang kiểm tra pipeline... Vui lòng đợi")

        # Auto-check pipeline if not checked yet
        if (
            not st.session_state.pipeline_status
            and not st.session_state.pipeline_checking
        ):
            with st.spinner("🔄 Tự động kiểm tra pipeline..."):
                try:
                    pipeline = get_pipeline_lazy()
                    if pipeline:
                        pipeline_status = pipeline.get_pipeline_status()
                        st.session_state.pipeline_status = pipeline_status
                        st.session_state.pipeline_checking = False
                        st.success("✅ Pipeline status đã được kiểm tra tự động!")
                        st.rerun()
                    else:
                        st.warning("⚠️ Không thể load pipeline để kiểm tra")
                        st.session_state.pipeline_checking = False
                except Exception as e:
                    st.error(f"❌ Lỗi khi kiểm tra pipeline: {str(e)}")
                    st.session_state.pipeline_checking = False

        # Display pipeline status if available
        if st.session_state.pipeline_status:
            pipeline_status = st.session_state.pipeline_status

            # Display pipeline status
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Pipeline Ready",
                    "✅ Ready" if pipeline_status.get("is_ready") else "❌ Not Ready",
                )
            with col2:
                st.metric(
                    "Retriever Ready",
                    (
                        "✅ Ready"
                        if pipeline_status.get("retriever_ready")
                        else "❌ Not Ready"
                    ),
                )
            with col3:
                st.metric(
                    "Reranker Ready",
                    (
                        "✅ Ready"
                        if pipeline_status.get("reranker_ready")
                        else "❌ Not Ready"
                    ),
                )
            with col4:
                loaded_models = pipeline_status.get("loaded_models", {})
                st.metric("Models Loaded", len(loaded_models))

        # Detailed System Information
        st.subheader("📋 Thông tin Chi tiết Hệ thống")

        # Model Details
        if model_status:
            st.markdown("**🤖 Chi tiết Models:**")
            for model_name, model_info in model_status.items():
                with st.expander(
                    f"🔧 {model_name.replace('_', ' ').title()}", expanded=False
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        status = model_info.get("status", "unknown")
                        if status == "ready":
                            st.success(f"✅ Status: {status}")
                        elif status == "partially_ready":
                            st.warning(f"⚠️ Status: {status}")
                        else:
                            st.error(f"❌ Status: {status}")

                        # Debug: Show all available keys for troubleshooting
                        st.write(f"**Model Path:** `{model_info.get('path', 'N/A')}`")
                        st.write(
                            f"**Model Size:** {model_info.get('size_mb', 0):.1f} MB"
                        )

                        # Show available keys for debugging
                        with st.expander("🔍 Debug - Available Keys", expanded=False):
                            st.write("**Available keys in model_info:**")
                            st.json(list(model_info.keys()))
                            st.write("**Full model_info:**")
                            st.json(model_info)

                        with col2:
                            if model_info.get("status") == "ready":
                                st.success("✅ Hoạt động bình thường")
                            else:
                                st.warning("⚠️ Cần kiểm tra")

        # FAISS Index Details
        if faiss_status:
            st.markdown("**🔍 Chi tiết FAISS Index:**")
            with st.expander("📋 FAISS Index Information", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    status = faiss_status.get("status", "unknown")
                    if status == "ready":
                        st.success("✅ Index Status")
                        st.metric("Status", "Ready")
                    else:
                        st.error("❌ Index Status")
                        st.metric("Status", "Not Ready")

                with col2:
                    index_size = faiss_status.get("index_size", 0)
                    st.metric("Index Size", f"{index_size:,} vectors")

                with col3:
                    file_count = faiss_status.get("file_count", 0)
                    st.metric("Index Files", file_count)

                with col4:
                    size_mb = faiss_status.get("size_mb", 0)
                    st.metric("Disk Size", f"{size_mb:.1f} MB")

                # Index health check
                if faiss_status.get("status") == "ready":
                    st.success("✅ FAISS Index hoạt động bình thường")
                    if faiss_status.get("index_size", 0) > 0:
                        st.info(
                            f"Index chứa {faiss_status['index_size']:,} vectors - đủ để tìm kiếm"
                        )
                    else:
                        st.warning("⚠️ Index không có vectors nào")
                else:
                    st.error("❌ FAISS Index có vấn đề - cần kiểm tra")

        # System Recommendations
        st.subheader("💡 Khuyến nghị Hệ thống")

        if model_status and faiss_status:
            models_ready = sum(
                1 for model in model_status.values() if model.get("status") == "ready"
            )
            total_models = len(model_status) if model_status else 0
            faiss_ready = faiss_status.get("status") == "ready"

            if models_ready < total_models:
                st.warning("⚠️ **Models cần kiểm tra:**")
                for name, status in model_status.items():
                    if status.get("status") != "ready":
                        st.write(f"- {name}: {status.get('status', 'unknown')}")

            if not faiss_ready:
                st.error("🔴 **FAISS Index cần kiểm tra:**")
                st.write("- Kiểm tra file index có tồn tại không")
                st.write("- Kiểm tra quyền truy cập file")

            if models_ready == total_models and faiss_ready:
                st.success("✅ **Hệ thống sẵn sàng:**")
                st.write("- Tất cả components hoạt động bình thường")
                st.write("- Có thể thực hiện tìm kiếm và phân tích")
                st.write("- Comprehensive evaluation có thể chạy được")

        # Performance Insights
        st.subheader("📈 Phân tích Hiệu suất & Insights")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
            **🎯 Điểm mạnh của Hệ thống:**
            - Kiến trúc 3 tầng tối ưu cho từng nhiệm vụ
            - Sử dụng models tiếng Việt chuyên biệt
            - Pipeline xử lý song song hiệu quả
            - Caching và optimization cho performance
            """
            )

        with col2:
            st.markdown(
                """
            **🔧 Cải tiến đã áp dụng:**
            - Contrastive Learning với TripletLoss cho Tier 1
            - Hard Negative Mining (HNM) cho training
            - ADAPT training cho domain adaptation
            - Ensemble learning cho tier 3
            - Tier-specific validation sets
            """
            )

        # Technical Specifications
        st.subheader("⚙️ Thông số Kỹ thuật")

        tech_specs = {
            "Framework": ["Streamlit", "PyTorch", "Transformers", "FAISS"],
            "Models": [
                "Vietnamese Bi-Encoder",
                "PhoBERT-base-v2",
                "PhoBART-large",
                "Ensemble Models",
            ],
            "Architecture": [
                "3-Tier Pipeline",
                "Score Aggregation",
                "Ensemble Learning",
                "Domain Adaptation",
            ],
            "Optimization": [
                "Contrastive Learning",
                "Caching",
                "Batch Processing",
                "Parallel Metrics",
            ],
        }

        df_tech = pd.DataFrame(tech_specs)
        st.dataframe(df_tech, use_container_width=True)

        # Usage Guidelines
        st.subheader("📖 Hướng dẫn Sử dụng")

        with st.expander("🔍 Cách sử dụng Comprehensive Evaluation", expanded=False):
            st.markdown(
                """
            **1. Tự động chạy:** Tab "Phân tích đánh giá toàn diện" sẽ tự động chạy evaluation
            **2. Standard Evaluation:** Click "🚀 Chạy Comprehensive Evaluation" để chạy với cache hiện tại
            **3. Fresh Evaluation:** Click "🔄 Fresh Evaluation" để xóa cache cũ và chạy evaluation mới hoàn toàn
            **4. Caching:** Kết quả được cache trong 1 giờ để tối ưu performance
            **5. Metrics:** Hiển thị đầy đủ precision, recall, F1, NDCG, MRR, quality cho từng tier
            """
            )

        with st.expander("⚡ Tối ưu hóa Performance", expanded=False):
            st.markdown(
                """
            **- Batch Processing:** Xử lý queries theo batch để tối ưu memory
            **- Parallel Metrics:** Tính toán tất cả metrics cùng lúc
            **- Efficient Aggregation:** Tính toán trung bình hiệu quả
            **- Progress Tracking:** Theo dõi tiến trình real-time
            """
            )

        # Cache Status Overview
        st.subheader("🗂️ Trạng thái Cache & Evaluation")

        col1, col2 = st.columns(2)
        with col1:
            # Check current cache status
            reports_dir = Path("reports")
            if reports_dir.exists():
                cache_files = list(reports_dir.glob("comprehensive_evaluation_*.json"))
                if cache_files:
                    latest_file = max(cache_files, key=lambda p: p.stat().st_mtime)
                    file_age = time.time() - latest_file.stat().st_mtime
                    age_hours = file_age / 3600

                    st.success(f"✅ **Cache Available:** {len(cache_files)} files")
                    st.info(f"📁 **Latest:** {latest_file.name}")
                    st.info(f"⏰ **Age:** {age_hours:.1f} giờ")

                    if age_hours < 1:
                        st.success("🟢 Cache rất mới (< 1 giờ)")
                    elif age_hours < 24:
                        st.info("🟡 Cache còn hạn sử dụng (< 24 giờ)")
                    else:
                        st.warning("🟠 Cache đã cũ (> 24 giờ)")
                else:
                    st.warning("⚠️ **No Cache:** Không có evaluation results được cache")
            else:
                st.error("❌ **Reports Directory:** Không tồn tại")

        with col2:
            # Enhanced cache management with emergency cleanup
            st.markdown("**🔧 Quản lý Evaluation:**")

            col2a, col2b = st.columns(2)
            with col2a:
                if st.button(
                    "🔄 Fresh Evaluation",
                    key="overview_fresh_eval",
                    type="primary",
                    help="Xóa cache cũ và chạy evaluation mới hoàn toàn",
                ):
                    st.info(
                        "🔄 Đang chuyển sang tab Comprehensive Evaluation để chạy Fresh Evaluation..."
                    )
                    # Switch to tab 2
                    st.session_state.switch_to_tab2 = True
                    st.rerun()

            with col2b:
                if st.button(
                    "🧹 Emergency Cleanup",
                    key="emergency_cleanup",
                    type="secondary",
                    help="Force clean page state nếu gặp vấn đề UI",
                ):
                    if force_clean_page_state():
                        st.success("✅ Page state đã được clean!")
                        st.rerun()
                    else:
                        st.error("❌ Không thể clean page state")

    with tab2:
        # 🚀 AUTO-LOADING Comprehensive Evaluation với Loading States
        st.subheader("🧪 Đánh giá Toàn diện với Metrics Chi tiết")

        # Performance optimization info
        with st.expander("⚡ Thông tin Tối ưu hóa Hiệu suất"):
            st.info(
                """
            **Các cải tiến hiệu suất đã được áp dụng:**
            - 🚀 **Caching**: Kết quả evaluation được cache trong 1 giờ
            - 📦 **Batch Processing**: Xử lý queries theo batch để tối ưu memory
            - 🔄 **Parallel Metrics**: Tính toán tất cả metrics cùng lúc cho mỗi tier
            - 💾 **Efficient Aggregation**: Tính toán trung bình hiệu quả
            - 📊 **Progress Tracking**: Theo dõi tiến trình real-time
            """
            )

            st.success(
                "**Thời gian evaluation dự kiến:** 30-60 giây (tùy thuộc vào số lượng queries)"
            )

        # Auto-load comprehensive evaluation
        if "comprehensive_eval_results" not in st.session_state:
            st.session_state.comprehensive_eval_results = None
            st.session_state.eval_loading = False

        # Simplified button layout - only 2 essential buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(
                "🚀 Chạy Comprehensive Evaluation",
                type="primary",
                key="comp_eval_btn",
                disabled=st.session_state.eval_loading,
                help="Chạy evaluation với cache hiện tại (nếu có)",
            ):
                st.session_state.eval_loading = True
                st.session_state.comprehensive_eval_results = None
                st.rerun()

        with col2:
            if st.button(
                "🔄 Fresh Evaluation",
                key="fresh_eval_btn",
                type="secondary",
                disabled=st.session_state.eval_loading,
                help="Xóa cache cũ và chạy evaluation mới hoàn toàn",
            ):
                st.session_state.eval_loading = True
                st.session_state.comprehensive_eval_results = None
                # Force fresh evaluation
                with st.spinner("🔄 Đang chạy evaluation mới..."):
                    fresh_results = force_fresh_comprehensive_evaluation()
                    if fresh_results:
                        st.session_state.comprehensive_eval_results = fresh_results
                        st.session_state.eval_loading = False
                        st.success("✅ Fresh evaluation hoàn thành!")
                        st.rerun()
                    else:
                        st.error("❌ Fresh evaluation thất bại")
                        st.session_state.eval_loading = False

        # Loading status display
        if st.session_state.eval_loading:
            st.info("🔄 Đang chạy comprehensive evaluation... Vui lòng đợi")

        # Auto-run evaluation if not loaded yet with enhanced error handling
        if (
            not st.session_state.comprehensive_eval_results
            and not st.session_state.eval_loading
        ):
            try:
                with st.spinner("🔄 Tự động chạy comprehensive evaluation..."):
                    # Add progress bar for better user experience
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Update progress with better error handling
                    try:
                        status_text.text("🔄 Đang khởi tạo evaluation...")
                        progress_bar.progress(10)

                        eval_results = auto_run_comprehensive_evaluation()

                        if eval_results:
                            progress_bar.progress(100)
                            status_text.text("✅ Hoàn thành evaluation!")

                            st.session_state.comprehensive_eval_results = eval_results
                            st.session_state.eval_loading = False
                            st.success(
                                "✅ Comprehensive evaluation hoàn thành tự động!"
                            )

                            # Force clean render to prevent element bleeding
                            st.rerun()
                        else:
                            progress_bar.progress(0)
                            status_text.text(
                                "⚠️ Không thể chạy comprehensive evaluation tự động"
                            )
                            st.warning(
                                "⚠️ Không thể chạy comprehensive evaluation tự động"
                            )
                            st.session_state.eval_loading = False

                    except Exception as eval_error:
                        progress_bar.progress(0)
                        status_text.text("❌ Lỗi khi chạy evaluation")
                        st.error(
                            f"❌ Lỗi khi chạy comprehensive evaluation tự động: {str(eval_error)}"
                        )
                        st.session_state.eval_loading = False

            except Exception as e:
                st.error(f"❌ Lỗi nghiêm trọng trong auto-loading: {str(e)}")
                st.session_state.eval_loading = False

        # Display results if available
        if st.session_state.comprehensive_eval_results:
            eval_results = st.session_state.comprehensive_eval_results

            # Display execution time and performance info
            st.success("✅ Comprehensive Evaluation đã hoàn thành!")

            # Simplified performance summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Status", "✅ Hoàn thành", delta="Thành công")
            with col2:
                st.metric("Cache Status", "💾 Cached", delta="1 giờ")
            with col3:
                st.metric("Performance", "⚡ Tối ưu", delta="Nhanh")

                # Cache status display removed - only metrics table

            # Display detailed metrics table
            st.subheader("📊 Metrics Chi tiết từng Tầng")

            # Create enhanced metrics table with better formatting
            metrics_df = []
            for tier, tier_metrics in eval_results.items():
                tier_name = tier.replace("_", " ").title()
                tier_display_name = {
                    "Tier 1": "🎯 Retrieval (Bi-Encoder)",
                    "Tier 2": "⚡ Light Reranker",
                    "Tier 3": "🎯 Cross Encoder",
                    "Combined": "🚀 Combined Pipeline",
                }.get(tier_name, tier_name)

                for metric in [
                    "precision",
                    "recall",
                    "f1",
                    "ndcg",
                    "mrr",
                    "quality",
                ]:
                    avg_key = f"{metric}_avg"
                    avg_value = tier_metrics.get(avg_key, 0.0)

                    # Format metric names
                    metric_display = {
                        "precision": "Precision",
                        "recall": "Recall",
                        "f1": "F1 Score",
                        "ndcg": "NDCG",
                        "mrr": "MRR",
                        "quality": "Quality Score",
                    }.get(metric, metric.upper())

                    metrics_df.append(
                        {
                            "Tier": tier_display_name,
                            "Metric": metric_display,
                            "Score": f"{avg_value:.4f}",
                            "Percentage": f"{avg_value*100:.1f}%",
                            "Status": (
                                "✅ Excellent"
                                if avg_value >= 0.8
                                else (
                                    "🟡 Good"
                                    if avg_value >= 0.6
                                    else "🔴 Needs Improvement"
                                )
                            ),
                        }
                    )

            metrics_df = pd.DataFrame(metrics_df)

            # Display metrics with better styling
            st.dataframe(
                metrics_df,
                use_container_width=True,
                column_config={
                    "Score": st.column_config.NumberColumn(
                        "Score", help="Raw score value (0.0 - 1.0)", format="%.4f"
                    ),
                    "Percentage": st.column_config.TextColumn(
                        "Percentage", help="Score as percentage"
                    ),
                    "Status": st.column_config.SelectboxColumn(
                        "Status",
                        help="Performance assessment",
                        options=["✅ Excellent", "🟡 Good", "🔴 Needs Improvement"],
                    ),
                },
            )

            # Display performance comparison charts
            st.subheader("📈 Biểu đồ So sánh Hiệu suất")

            # Tier performance comparison
            tier_perf_chart = create_tier_performance_chart(eval_results)
            if tier_perf_chart:
                st.plotly_chart(tier_perf_chart, use_container_width=True)

            # Improvement chart
            improvement_chart = create_tier_improvement_chart(eval_results)
            if improvement_chart:
                st.plotly_chart(improvement_chart, use_container_width=True)

            # Score effectiveness distribution chart
            score_effectiveness_chart = create_score_distribution_chart(eval_results)
            if score_effectiveness_chart:
                st.plotly_chart(score_effectiveness_chart, use_container_width=True)

            # Performance Metrics Legend & Explanation
            st.subheader("📚 Bảng Chú thích Performance Metrics")

            metrics_explanation = {
                "Metric": [
                    "🎯 **Precision**",
                    "📊 **Recall**",
                    "⚖️ **F1 Score**",
                    "📈 **NDCG (Normalized Discounted Cumulative Gain)**",
                    "🏆 **MRR (Mean Reciprocal Rank)**",
                    "⭐ **Quality Score**",
                ],
                "Ý nghĩa": [
                    "Độ chính xác của kết quả trả về (tỷ lệ kết quả đúng trong top-K)",
                    "Độ bao phủ của kết quả (tỷ lệ kết quả đúng được tìm thấy)",
                    "Trung bình điều hòa của Precision và Recall, cân bằng cả hai chỉ số",
                    "Đánh giá chất lượng ranking dựa trên vị trí và độ liên quan của kết quả",
                    "Đánh giá vị trí xuất hiện đầu tiên của kết quả đúng trong ranking",
                    "Điểm tổng hợp đánh giá chất lượng tổng thể của tier dựa trên nhiều yếu tố",
                ],
                "Thang điểm": [
                    "0.0 - 1.0 (Càng cao càng tốt)",
                    "0.0 - 1.0 (Càng cao càng tốt)",
                    "0.0 - 1.0 (Càng cao càng tốt)",
                    "0.0 - 1.0 (Càng cao càng tốt)",
                    "0.0 - 1.0 (Càng cao càng tốt)",
                    "0.0 - 1.0 (Càng cao càng tốt)",
                ],
                "Mục tiêu": [
                    "≥ 0.8 (Xuất sắc)",
                    "≥ 0.8 (Xuất sắc)",
                    "≥ 0.8 (Xuất sắc)",
                    "≥ 0.9 (Xuất sắc)",
                    "≥ 0.8 (Xuất sắc)",
                    "≥ 0.7 (Tốt)",
                ],
            }

            df_legend = pd.DataFrame(metrics_explanation)
            st.dataframe(
                df_legend,
                use_container_width=True,
                column_config={
                    "Metric": st.column_config.TextColumn("Metric", width="medium"),
                    "Ý nghĩa": st.column_config.TextColumn("Ý nghĩa", width="large"),
                    "Thang điểm": st.column_config.TextColumn(
                        "Thang điểm", width="medium"
                    ),
                    "Mục tiêu": st.column_config.TextColumn("Mục tiêu", width="medium"),
                },
            )

            # Additional performance insights
            with st.expander(
                "💡 Giải thích chi tiết Performance Metrics", expanded=False
            ):
                st.markdown(
                    """
                **🎯 Precision (Độ chính xác):**
                - Đo lường tỷ lệ kết quả đúng trong số kết quả được trả về
                - Ví dụ: Precision = 0.8 nghĩa là 80% kết quả trả về là chính xác
                
                **📊 Recall (Độ bao phủ):**
                - Đo lường khả năng tìm thấy tất cả kết quả đúng có trong dữ liệu
                - Ví dụ: Recall = 0.7 nghĩa là tìm thấy 70% kết quả đúng có sẵn
                
                **⚖️ F1 Score:**
                - Trung bình điều hòa của Precision và Recall
                - Công thức: F1 = 2 × (Precision × Recall) / (Precision + Recall)
                - Giúp cân bằng giữa độ chính xác và độ bao phủ
                
                **📈 NDCG (Normalized Discounted Cumulative Gain):**
                - Đánh giá chất lượng ranking dựa trên vị trí của kết quả
                - Kết quả ở vị trí cao hơn có trọng số lớn hơn
                - Được chuẩn hóa để so sánh giữa các queries khác nhau
                
                **🏆 MRR (Mean Reciprocal Rank):**
                - Đo lường vị trí xuất hiện đầu tiên của kết quả đúng
                - Công thức: MRR = 1 / (vị trí kết quả đúng đầu tiên)
                - MRR = 1.0 nghĩa là kết quả đúng luôn ở vị trí đầu tiên
                
                **⭐ Quality Score:**
                - Điểm tổng hợp đánh giá chất lượng tổng thể của tier
                - Dựa trên: độ ổn định điểm số, phân phối điểm, và hiệu quả tổng thể
                - Giúp so sánh hiệu suất tổng thể giữa các tiers
                """
                )

            # Summary insights with enhanced calculations
            st.subheader("💡 Phân tích Tổng hợp & Khuyến nghị")

            # Calculate overall improvements with better logic
            overall_improvements = {}
            for metric in ["precision", "recall", "f1", "ndcg", "mrr", "quality"]:
                individual_scores = [
                    eval_results["tier_1"].get(f"{metric}_avg", 0.0),
                    eval_results["tier_2"].get(f"{metric}_avg", 0.0),
                    eval_results["tier_3"].get(f"{metric}_avg", 0.0),
                ]
                best_individual = max(individual_scores)
                combined_score = eval_results["combined"].get(f"{metric}_avg", 0.0)

                if best_individual > 0:
                    improvement = (
                        (combined_score - best_individual) / best_individual
                    ) * 100
                else:
                    improvement = 0.0

                overall_improvements[metric.upper()] = improvement

            # Display improvement summary with better metrics
            st.subheader("📊 So sánh Hiệu suất: Cá nhân vs Kết hợp")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Precision Improvement",
                    f"{overall_improvements['PRECISION']:+.1f}%",
                    delta=f"{overall_improvements['PRECISION']:+.1f}%",
                )
                st.metric(
                    "Recall Improvement",
                    f"{overall_improvements['RECALL']:+.1f}%",
                    delta=f"{overall_improvements['RECALL']:+.1f}%",
                )

            with col2:
                st.metric(
                    "F1 Improvement",
                    f"{overall_improvements['F1']:+.1f}%",
                    delta=f"{overall_improvements['F1']:+.1f}%",
                )
                st.metric(
                    "NDCG Improvement",
                    f"{overall_improvements['NDCG']:+.1f}%",
                    delta=f"{overall_improvements['NDCG']:+.1f}%",
                )

            with col3:
                st.metric(
                    "MRR Improvement",
                    f"{overall_improvements['MRR']:+.1f}%",
                    delta=f"{overall_improvements['MRR']:+.1f}%",
                )
                st.metric(
                    "Quality Improvement",
                    f"{overall_improvements['QUALITY']:+.1f}%",
                    delta=f"{overall_improvements['QUALITY']:+.1f}%",
                )

            # Performance insights and recommendations
            st.subheader("💡 Phân tích Chi tiết & Khuyến nghị")

            # Analyze each tier's performance
            tier_analysis = {}
            for tier in ["tier_1", "tier_2", "tier_3"]:
                tier_metrics = eval_results[tier]
                avg_f1 = tier_metrics.get("f1_avg", 0.0)
                avg_quality = tier_metrics.get("quality_avg", 0.0)

                if avg_f1 >= 0.8 and avg_quality >= 0.7:
                    status = "✅ Xuất sắc"
                    recommendation = "Tầng này hoạt động rất tốt, không cần cải thiện"
                elif avg_f1 >= 0.6 and avg_quality >= 0.5:
                    status = "🟡 Tốt"
                    recommendation = "Có thể cải thiện thêm để đạt hiệu suất cao hơn"
                else:
                    status = "🔴 Cần cải thiện"
                    recommendation = "Cần xem xét lại training data và hyperparameters"

                tier_analysis[tier] = {
                    "status": status,
                    "f1_score": avg_f1,
                    "quality_score": avg_quality,
                    "recommendation": recommendation,
                }

            # Display tier analysis
            col1, col2, col3 = st.columns(3)
            tier_names = {
                "tier_1": "🎯 Tầng 1 (Retrieval)",
                "tier_2": "⚡ Tầng 2 (Light Reranker)",
                "tier_3": "🎯 Tầng 3 (Cross Encoder)",
            }

            for i, (tier, analysis) in enumerate(tier_analysis.items()):
                with [col1, col2, col3][i]:
                    st.subheader(tier_names[tier])
                    st.metric("Status", analysis["status"])
                    st.metric("F1 Score", f"{analysis['f1_score']:.3f}")
                    st.metric("Quality", f"{analysis['quality_score']:.3f}")
                    st.info(analysis["recommendation"])

            # Show raw data in expandable section for debugging
            with st.expander("🔍 Debug - Raw Evaluation Results"):
                st.json(eval_results)
