#!/usr/bin/env python3
"""
LegalQA Pipeline - LawBot v8.1 (Refactored)
===========================================

Main pipeline for legal question answering that orchestrates retrieval and
reranking components using a simplified, unified architecture.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import sys

try:
    from core.retrieval import RetrievalEngine
    from core.reranking import RerankingEngine
    from core.utils.logging_manager import get_logger
    from core.utils.versioning import get_latest_version_path
    from core.utils.parent_law_manager import ensure_parent_law_mapping
    from config.loader import config
except ImportError as e:
    logging.error(f"Import error in pipeline.py: {e}")
    raise

logger = get_logger(__name__)


class LegalQAPipeline:
    """LegalQA Pipeline class for orchestrating retrieval and reranking."""

    def __init__(
        self,
        bi_encoder_path: Optional[Path] = None,
        reranker_paths: Optional[Dict[str, Path]] = None,
        faiss_index_path: Optional[Path] = None,
        content_map_path: Optional[Path] = None,
        index_to_aid_path: Optional[Path] = None,
    ):
        """Initialize the LegalQA Pipeline."""
        self.is_ready = False
        self.loaded_model_paths = {}

        try:
            # Ensure parent law mapping is available before proceeding
            logger.info("Validating parent law mapping availability...")
            if not ensure_parent_law_mapping():
                logger.warning(
                    "Parent law mapping validation failed, but continuing with pipeline initialization"
                )
            else:
                logger.info("✅ Parent law mapping validation successful")

            # --- Load Retriever ---
            paths = config.paths
            bi_encoder_to_load = bi_encoder_path or get_latest_version_path(
                paths.model_dir, "bi-encoder"
            )
            faiss_to_load = faiss_index_path or paths.faiss_index_path
            content_map_to_load = content_map_path or paths.content_map_path
            index_to_aid_to_load = index_to_aid_path or paths.index_to_aid_path

            if not all(
                [
                    bi_encoder_to_load,
                    faiss_to_load.exists(),
                    content_map_to_load.exists(),
                    index_to_aid_to_load.exists(),
                ]
            ):
                raise FileNotFoundError(
                    f"One or more required files for the retriever could not be found. "
                    f"Checked paths: Bi-Encoder: {bi_encoder_to_load}, "
                    f"FAISS: {faiss_to_load}, Content Map: {content_map_to_load}, "
                    f"Index Map: {index_to_aid_to_load}"
                )

            logger.info(f"Loading Retriever with Bi-Encoder: {bi_encoder_to_load}")
            self.retriever = RetrievalEngine(
                bi_encoder_path=str(bi_encoder_to_load),
                faiss_index_path=str(faiss_to_load),
                content_map_path=str(content_map_to_load),
                index_to_aid_path=str(index_to_aid_to_load),
            )
            self.loaded_model_paths["bi_encoder"] = bi_encoder_to_load
            self.loaded_model_paths["faiss_index"] = faiss_to_load

            # --- Load Unified Reranker ---
            reranker_configs_to_load = self._resolve_reranker_paths(reranker_paths)
            if not reranker_configs_to_load:
                logger.warning(
                    "No reranker models configured, continuing without reranking."
                )
                self.reranker = None
            else:
                self.reranker = RerankingEngine(
                    reranker_configs=reranker_configs_to_load
                )
                self.loaded_model_paths["rerankers"] = {
                    name: cfg["path"] for name, cfg in reranker_configs_to_load.items()
                }

            self.is_ready = True
            logger.info("LegalQAPipeline initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize LegalQAPipeline: {e}", exc_info=True)
            self.is_ready = False
            raise

    def get_loaded_model_versions(self) -> Dict[str, str]:
        """Returns a dictionary of loaded model names and their version IDs."""
        versions = {}
        if "bi_encoder" in self.loaded_model_paths:
            versions["Bi-Encoder"] = Path(self.loaded_model_paths["bi_encoder"]).name
        if "rerankers" in self.loaded_model_paths:
            for name, path in self.loaded_model_paths["rerankers"].items():
                versions[name.replace("_", " ").title()] = Path(path).name
        return versions

    def _resolve_reranker_paths(
        self, manual_paths: Optional[Dict[str, Path]]
    ) -> Dict[str, Any]:
        """Resolves paths for reranker models."""
        if manual_paths:
            logger.info(
                f"Loading rerankers from manually provided paths: {manual_paths}"
            )
            return {
                name: {"path": path, "enabled": True}
                for name, path in manual_paths.items()
            }

        logger.info("Auto-discovering latest reranker model versions...")
        resolved_configs = {}

        # Map config names to actual model directory names
        reranker_model_mapping = {
            "light_reranker": "light-ranking",  # SentenceTransformer model
            "cross_encoder": "combined-reranker-adapt",  # Ensemble model
        }

        # Map config names to actual model subdirectories (for models with subdirectory structure)
        # Note: cross_encoder is an ensemble model loaded from root directory
        reranker_subdir_mapping = {
            # "cross_encoder": "adapt_enhanced",  # Cross-encoder model is loaded from root directory
        }

        reranker_models = {
            "light_reranker": config.reranker_pipeline.light_reranker,
            "cross_encoder": config.reranker_pipeline.cross_encoder,
        }

        for config_name, model_cfg in reranker_models.items():
            if (
                model_cfg
                and isinstance(model_cfg, dict)
                and model_cfg.get("enabled", False)
            ):
                # Get the actual model directory name
                model_dir_name = reranker_model_mapping.get(config_name, config_name)
                latest_path = get_latest_version_path(
                    config.paths.model_dir, model_dir_name
                )
                if latest_path:
                    # Check if this model has a subdirectory structure
                    if config_name in reranker_subdir_mapping:
                        subdir = reranker_subdir_mapping[config_name]
                        model_path = latest_path / subdir
                        if model_path.exists():
                            latest_path = model_path
                            logger.info(
                                f"Using subdirectory {subdir} for {config_name}: {model_path}"
                            )
                        else:
                            logger.warning(
                                f"Subdirectory {subdir} not found for {config_name} at {model_path}"
                            )

                    # Create a dictionary from the config
                    resolved_configs[config_name] = model_cfg.copy()
                    resolved_configs[config_name]["path"] = str(latest_path)

                    # Add model type information for proper handling
                    if config_name == "light_reranker":
                        resolved_configs[config_name][
                            "model_type"
                        ] = "sentence_transformer"
                    else:
                        resolved_configs[config_name]["model_type"] = "classification"

                    logger.info(
                        f"Found reranker {config_name} ({model_dir_name}) at: {latest_path}"
                    )
                else:
                    logger.warning(
                        f"Reranker {config_name} ({model_dir_name}) is enabled but no model found"
                    )
            else:
                logger.info(f"Reranker {config_name} is disabled or not configured")
        return resolved_configs

    def predict(
        self,
        query: str,
        top_k_retrieval: Optional[int] = None,
        top_k_light: Optional[int] = None,  # ✅ Add top_k_light parameter
        top_k_final: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Predict relevant documents using the 3-tier architecture.
        Parameters are taken from the config file but can be overridden.
        """
        # Use config values as defaults if parameters are not provided
        cfg_app = config.app
        cfg_reranker = config.reranker_pipeline

        top_k_retrieval = top_k_retrieval or cfg_app.top_k_retrieval
        top_k_final = top_k_final or cfg_app.top_k_final

        use_light_ranking = cfg_reranker.light_reranker.get("enabled", False)
        # ✅ Use top_k_light from params if provided, otherwise from config
        top_k_light = top_k_light or cfg_reranker.light_reranker.get("top_k", 80)
        light_weight = cfg_reranker.light_reranker.get("weight", 0.7)

        use_cross_encoder = cfg_reranker.cross_encoder.get("enabled", False)
        cross_encoder_weight = cfg_reranker.cross_encoder.get("weight", 0.3)

        # Debug logging for reranker configuration
        logger.info(
            f"🔧 Reranker Config - Light: {use_light_ranking}, Cross: {use_cross_encoder}"
        )
        logger.info(
            f"🔧 Reranker Weights - Light: {light_weight}, Cross: {cross_encoder_weight}"
        )
        logger.info(
            f"🔧 Reranker Engine Ready: {self.reranker.is_ready if self.reranker else False}"
        )

        if not self.is_ready:
            raise RuntimeError("Pipeline is not ready. Please check initialization.")

        try:
            logger.info(f"Processing query: {query[:100]}...")
            logger.info(
                f"Tiers: Ret({top_k_retrieval}) -> Light({top_k_light}) -> Cross -> Final({top_k_final})"
            )
            logger.info(
                f"🔧 Parameters: retrieval={top_k_retrieval}, light={top_k_light}, final={top_k_final}"
            )

            # Tier 1: Bi-Encoder Retrieval
            logger.info("🎯 Tier 1: Bi-Encoder Retrieval")
            documents = self.retriever.retrieve(query, top_k=top_k_retrieval)
            if not documents:
                logger.warning("No documents retrieved from Bi-Encoder")
                return []

            # Initialize scores
            for doc in documents:
                doc["light_reranker_score"] = 0.0
                doc["cross_encoder_score"] = 0.0
                # Debug logging
                # logger.info(
                #     f"🔍 Document initialized with scores: light={doc['light_reranker_score']}, cross={doc['cross_encoder_score']}"
                # )

            # Tier 2: Light Reranking (if enabled)
            if use_light_ranking and self.reranker and self.reranker.is_ready:
                logger.info("⚡ Tier 2: Light Reranking")
                documents = self.reranker.rank_light(query, documents[:top_k_light])
                # Debug: Check if scores were updated
                for doc in documents[:3]:  # Check first 3 docs
                    logger.info(
                        f"🔍 After light reranking: light_score={doc.get('light_reranker_score', 'N/A')}"
                    )
            else:
                logger.warning(
                    f"⚠️ Light reranking skipped: use_light_ranking={use_light_ranking}, reranker={self.reranker is not None}, is_ready={self.reranker.is_ready if self.reranker else False}"
                )

            # Tier 3: Cross-Encoder Reranking (if enabled)
            if use_cross_encoder and self.reranker and self.reranker.is_ready:
                logger.info("🎯 Tier 3: Cross-Encoder Reranking")
                # Rerank the top candidates from the previous stage
                documents = self.reranker.rank_cross(query, documents)
                # Debug: Check if scores were updated
                for doc in documents[:3]:  # Check first 3 docs
                    logger.info(
                        f"🔍 After cross-encoder reranking: cross_score={doc.get('cross_encoder_score', 'N/A')}"
                    )
            else:
                logger.warning(
                    f"⚠️ Cross-encoder reranking skipped: use_cross_encoder={use_cross_encoder}, reranker={self.reranker is not None}, is_ready={self.reranker.is_ready if self.reranker else False}"
                )

            # Combine scores and sort
            logger.info("🔄 Combining scores from all tiers...")
            for doc in documents:
                retrieval_score = doc.get("retrieval_score", 0.0)
                light_score = doc.get("light_reranker_score", 0.0)
                cross_score = doc.get("cross_encoder_score", 0.0)

                # Flexible score combination based on which tiers were used
                final_score = retrieval_score  # Start with base score
                if use_light_ranking:
                    final_score = (
                        light_weight * light_score + (1 - light_weight) * final_score
                    )
                if use_cross_encoder:
                    final_score = (
                        cross_encoder_weight * cross_score
                        + (1 - cross_encoder_weight) * final_score
                    )

                doc["final_score"] = final_score
                doc["score_breakdown"] = {
                    "retrieval": retrieval_score,
                    "light_reranker": light_score,
                    "cross_encoder": cross_score,
                }

            final_results = sorted(
                documents, key=lambda x: x["final_score"], reverse=True
            )[:top_k_final]
            logger.info(
                f"✅ Final ranking completed. Returning {len(final_results)} results"
            )

            return final_results

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return []

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get the current status of the pipeline."""
        status = {
            "is_ready": self.is_ready,
            "loaded_models": self.get_loaded_model_versions(),
            "retriever_ready": hasattr(self, "retriever") and self.retriever.is_ready,
            "reranker_ready": False,
        }
        if hasattr(self, "reranker") and self.reranker is not None:
            status["reranker_ready"] = self.reranker.is_ready
            status["reranker_info"] = self.reranker.get_engine_info()
        return status

    def cleanup(self):
        """Clean up resources used by the pipeline."""
        try:
            if hasattr(self, "retriever"):
                self.retriever.cleanup()
            if hasattr(self, "reranker"):
                self.reranker.cleanup()
            logger.info("Pipeline cleanup completed")
        except Exception as e:
            logger.warning(f"Error during pipeline cleanup: {e}")


# Alias for backward compatibility
Pipeline = LegalQAPipeline
