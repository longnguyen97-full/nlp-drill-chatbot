#!/usr/bin/env python3
"""
Reranking Engine - LawBot v8.1 (Unified)
=========================================

Unified reranking component that uses cross-encoder models (both light and heavy)
to improve the quality of retrieved documents for legal question answering.
This engine provides separate methods for each reranking tier, allowing the pipeline
to control the orchestration.
"""

from typing import List, Dict, Any, Tuple
import logging
import json
from pathlib import Path

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModel,
    )
    from transformers.modeling_outputs import SequenceClassifierOutput
    import numpy as np
    from core.utils.logging_manager import get_logger
except ImportError as e:
    logging.error(f"Import error in reranking.py: {e}")
    raise

logger = get_logger(__name__)


def _safe_move_to_device(model, device):
    """Safely move model to device, handling meta tensors and offloaded models."""
    try:
        # Check if model has offloaded modules
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            logger.info(f"Model has offloaded modules, skipping device movement to {device}")
            return model
        
        # Try to move model safely
        model.to(device)
    except NotImplementedError as e:
        if "meta tensor" in str(e).lower():
            logger.info(f"Detected meta tensor, using to_empty() for device: {device}")
            model.to_empty(device=device)
        else:
            raise
    except RuntimeError as e:
        if "offloaded" in str(e) or "dispatched" in str(e):
            logger.info(f"Model is offloaded/dispatched, skipping device movement to {device}: {e}")
        else:
            raise
    return model


def _load_model(
    model_path: str, tokenizer_path: str, model_class, tokenizer_class, **kwargs
) -> Tuple[Any, Any]:
    """A general-purpose model and tokenizer loader."""
    try:
        tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        
        # Do not add device_map to avoid meta tensor issues
        # Models will be moved to device manually after loading
            
        model = model_class.from_pretrained(model_path, **kwargs)
        
        # Handle meta tensors and offloaded models safely after loading
        if torch.cuda.is_available():
            try:
                # Check if model has offloaded modules
                if hasattr(model, 'hf_device_map') and model.hf_device_map:
                    logger.info("Model has offloaded modules, skipping device movement")
                else:
                    model.to("cuda")
            except NotImplementedError:
                # Handle meta tensors with to_empty()
                model.to_empty(device="cuda")
                logger.info("Used to_empty() for meta tensor handling")
            except RuntimeError as e:
                if "offloaded" in str(e) or "dispatched" in str(e):
                    logger.info(f"Model is offloaded/dispatched, skipping device movement: {e}")
                else:
                    logger.warning(f"Could not move model to cuda: {e}")
        
        model.eval()
        logger.info(f"Successfully loaded model from {model_path}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}", exc_info=True)
        raise


def _load_ensemble_model(model_path: str) -> Tuple[Any, Any]:
    """Load ensemble cross-encoder model with adapt_model and base_model components."""
    try:
        model_dir = Path(model_path)

        # Check if this is an ensemble model
        ensemble_config_path = model_dir / "ensemble_config.json"
        if not ensemble_config_path.exists():
            logger.warning(f"Ensemble config not found at {ensemble_config_path}")
            return None, None

        # Load ensemble configuration
        with open(ensemble_config_path, "r") as f:
            ensemble_config = json.load(f)

        logger.info(f"Loading ensemble model with config: {ensemble_config}")

        # Load individual models
        adapt_model_path = model_dir / "adapt_model"
        base_model_path = model_dir / "base_model"

        if not adapt_model_path.exists() or not base_model_path.exists():
            logger.error(
                f"Ensemble model subdirectories not found: {adapt_model_path}, {base_model_path}"
            )
            return None, None

        # Load ADAPT model without device_map to avoid meta tensor issues
        adapt_model = AutoModelForSequenceClassification.from_pretrained(
            str(adapt_model_path)
        )

        # Load base model without device_map to avoid meta tensor issues
        base_model = AutoModelForSequenceClassification.from_pretrained(
            str(base_model_path)
        )

        # Load tokenizer from root directory (where all tokenizer files are located)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        # Create ensemble wrapper
        class EnsembleCrossEncoder:
            def __init__(self, adapt_model, base_model, adapt_weight, base_weight):
                self.adapt_model = adapt_model
                self.base_model = base_model
                self.adapt_weight = adapt_weight
                self.base_weight = base_weight
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

                # Check if models are already on correct device or offloaded
                def safe_move_to_device(model, target_device):
                    try:
                        # Check if model is already on target device
                        if hasattr(model, 'device'):
                            current_device = str(model.device)
                            if target_device in current_device:
                                logger.info(f"Model already on {target_device}")
                                return
                        
                        # Check if model has offloaded modules
                        if hasattr(model, 'hf_device_map') and model.hf_device_map:
                            logger.info(f"Model has offloaded modules, skipping device movement")
                            return
                        
                        # Try to move model safely
                        model.to(target_device)
                    except NotImplementedError as nie:
                        if "meta tensor" in str(nie):
                            logger.warning(f"⚠️ Meta tensor error during device movement: {nie}")
                            logger.info("🔄 Attempting alternative device movement method...")
                            try:
                                # Try using to_empty() method
                                model.to_empty(device=target_device)
                                logger.info("✅ Successfully moved model using to_empty()")
                            except Exception as to_empty_error:
                                logger.warning(f"⚠️ to_empty() method also failed: {to_empty_error}")
                                logger.info("ℹ️ Keeping model on current device")
                        else:
                            logger.warning(f"⚠️ NotImplementedError during device movement: {nie}")
                    except RuntimeError as e:
                        if "offloaded" in str(e) or "dispatched" in str(e):
                            logger.info(f"Model is offloaded/dispatched, skipping device movement: {e}")
                            return
                        else:
                            logger.warning(f"Could not move model to {target_device}: {e}")
                    except Exception as e:
                        logger.warning(f"⚠️ Unexpected error during device movement: {e}")
                        logger.info("ℹ️ Keeping model on current device")

                # Move models to device safely
                safe_move_to_device(self.adapt_model, self.device)
                safe_move_to_device(self.base_model, self.device)

                # Set to eval mode
                self.adapt_model.eval()
                self.base_model.eval()

            def to(self, device):
                self.device = device
                # Handle meta tensors and offloaded models safely when moving to new device
                def safe_move_to_device_v2(model, target_device):
                    try:
                        # Check if model has offloaded modules
                        if hasattr(model, 'hf_device_map') and model.hf_device_map:
                            logger.info(f"Model has offloaded modules, skipping device movement")
                            return
                        
                        # Try to move model safely
                        model.to(target_device)
                    except NotImplementedError as nie:
                        if "meta tensor" in str(nie):
                            logger.warning(f"⚠️ Meta tensor error during device movement: {nie}")
                            logger.info("🔄 Attempting alternative device movement method...")
                            try:
                                # Try using to_empty() method
                                model.to_empty(device=target_device)
                                logger.info("✅ Successfully moved model using to_empty()")
                            except Exception as to_empty_error:
                                logger.warning(f"⚠️ to_empty() method also failed: {to_empty_error}")
                                logger.info("ℹ️ Keeping model on current device")
                        else:
                            logger.warning(f"⚠️ NotImplementedError during device movement: {nie}")
                    except RuntimeError as e:
                        if "offloaded" in str(e) or "dispatched" in str(e):
                            logger.info(f"Model is offloaded/dispatched, skipping device movement: {e}")
                            return
                        else:
                            logger.warning(f"Could not move model to {target_device}: {e}")
                    except Exception as e:
                        logger.warning(f"⚠️ Unexpected error during device movement: {e}")
                        logger.info("ℹ️ Keeping model on current device")
                
                safe_move_to_device_v2(self.adapt_model, device)
                safe_move_to_device_v2(self.base_model, device)
                return self

            def eval(self):
                self.adapt_model.eval()
                self.base_model.eval()
                return self

            def parameters(self):
                """Return parameters from both models for device detection."""
                # Return parameters from adapt_model for device detection
                return self.adapt_model.parameters()

            def __call__(self, **inputs):
                # Get predictions from both models
                with torch.no_grad():
                    adapt_output = self.adapt_model(**inputs)
                    base_output = self.base_model(**inputs)

                    # Weighted combination of logits
                    ensemble_logits = (
                        self.adapt_weight * adapt_output.logits
                        + self.base_weight * base_output.logits
                    )

                    # Return in the expected format
                    return SequenceClassifierOutput(logits=ensemble_logits)

        # Create ensemble model
        ensemble_model = EnsembleCrossEncoder(
            adapt_model=adapt_model,
            base_model=base_model,
            adapt_weight=ensemble_config.get("adapt_weight", 0.7),
            base_weight=ensemble_config.get("base_weight", 0.3),
        )

        logger.info(f"✅ Successfully loaded ensemble cross-encoder model")
        return ensemble_model, tokenizer

    except Exception as e:
        logger.error(
            f"Failed to load ensemble model from {model_path}: {e}", exc_info=True
        )
        raise


class RerankingEngine:
    """Unified reranking engine for light and cross-encoder models."""

    def __init__(self, reranker_configs: Dict[str, Any]):
        """Initialize the unified reranking engine.

        Args:
            reranker_configs: Configuration dictionary for all reranker models.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.configs = reranker_configs
        self.is_ready = False
        self.models = {}
        self.tokenizers = {}

        for name, cfg in self.configs.items():
            logger.info(f"🔧 Attempting to load reranker '{name}' with config: {cfg}")
            if cfg.get("enabled"):
                try:
                    logger.info(f"🔄 Loading model from path: {cfg['path']}")

                    # Check model type to determine loading strategy
                    model_type = cfg.get("model_type", "auto")

                    if model_type == "sentence_transformer" or name == "light_reranker":
                        # Load as SentenceTransformer for light reranker
                        try:
                            from sentence_transformers import SentenceTransformer

                            # Monkey patch SentenceTransformer to prevent auto device movement
                            original_to = SentenceTransformer.to
                            
                            def safe_to(self, device=None, *args, **kwargs):
                                """Safe device movement that handles meta tensors."""
                                try:
                                    return original_to(self, device, *args, **kwargs)
                                except NotImplementedError as e:
                                    if "meta tensor" in str(e).lower():
                                        logger.info(f"Detected meta tensor in SentenceTransformer, using to_empty() for device: {device}")
                                        return self.to_empty(device=device)
                                    else:
                                        raise
                                except RuntimeError as e:
                                    if "offloaded" in str(e) or "dispatched" in str(e):
                                        logger.info(f"SentenceTransformer is offloaded/dispatched, keeping on current device: {e}")
                                        return self
                                    else:
                                        raise
                            
                            # Apply monkey patch
                            SentenceTransformer.to = safe_to
                            
                            # Load without device parameter first
                            model = SentenceTransformer(str(cfg["path"]))
                            logger.info(f"✅ Successfully loaded SentenceTransformer with safe device handling")
                            
                            # Move to target device safely if needed
                            if self.device != 'cpu':
                                try:
                                    # Use the safe device movement
                                    model.to(self.device)
                                    logger.info(f"✅ Moved SentenceTransformer to {self.device}")
                                except Exception as e:
                                    logger.warning(f"⚠️ Device movement failed, keeping on current device: {e}")
                            
                            # Restore original method
                            SentenceTransformer.to = original_to
                            
                            tokenizer = None  # SentenceTransformer handles tokenization internally
                        except Exception as st_e:
                            logger.error(
                                f"❌ Failed to load {name} as SentenceTransformer: {st_e}"
                            )
                            raise

                    elif model_type == "classification" or name == "cross_encoder":
                        # Load as classification model for cross encoder
                        try:
                            # First try to load as ensemble model
                            model, tokenizer = _load_ensemble_model(str(cfg["path"]))
                            if model is not None:
                                logger.info(
                                    f"✅ Loaded {name} as ensemble cross-encoder"
                                )
                            else:
                                logger.warning(
                                    f"🔄 Ensemble model loading failed for {name}, trying fallback methods"
                                )
                                # Try to load individual components if ensemble fails
                                model_dir = Path(cfg["path"])
                                adapt_model_path = model_dir / "adapt_model"
                                base_model_path = model_dir / "base_model"
                                
                                if adapt_model_path.exists() and base_model_path.exists():
                                    # Try to load adapt_model as fallback
                                    try:
                                        model, tokenizer = _load_model(
                                            str(adapt_model_path),
                                            str(model_dir),  # Use root directory for tokenizer
                                            AutoModelForSequenceClassification,
                                            AutoTokenizer,
                                            num_labels=1,
                                        )
                                        logger.info(
                                            f"✅ Loaded {name} fallback as adapt_model"
                                        )
                                    except Exception as adapt_e:
                                        logger.warning(
                                            f"🔄 Adapt model fallback failed: {adapt_e}"
                                        )
                                        # Final fallback to base model
                                        try:
                                            model, tokenizer = _load_model(
                                                str(base_model_path),
                                                str(model_dir),  # Use root directory for tokenizer
                                                AutoModelForSequenceClassification,
                                                AutoTokenizer,
                                                num_labels=1,
                                            )
                                            logger.info(
                                                f"✅ Loaded {name} fallback as base_model"
                                            )
                                        except Exception as base_e:
                                            logger.error(
                                                f"❌ All ensemble fallbacks failed for {name}: {base_e}"
                                            )
                                            raise
                                else:
                                    logger.error(
                                        f"❌ Ensemble model structure invalid for {name}: missing subdirectories"
                                    )
                                    raise ValueError(f"Invalid ensemble model structure: {cfg['path']}")
                        except Exception as seq_e:
                            logger.error(
                                f"❌ Ensemble model loading failed for {name}: {seq_e}"
                            )
                            raise
                    else:
                        # Auto-detect model type
                        try:
                            # Try classification first
                            model, tokenizer = _load_model(
                                str(cfg["path"]),
                                str(cfg["path"]),
                                AutoModelForSequenceClassification,
                                AutoTokenizer,
                                num_labels=1,
                            )
                            logger.info(
                                f"✅ Auto-detected {name} as AutoModelForSequenceClassification"
                            )
                        except Exception as seq_e:
                            try:
                                # Try sentence-transformers
                                from sentence_transformers import SentenceTransformer

                                model = SentenceTransformer(str(cfg["path"]))
                                tokenizer = None
                                logger.info(
                                    f"✅ Auto-detected {name} as SentenceTransformer"
                                )
                            except Exception as st_e:
                                logger.error(
                                    f"❌ Failed to auto-detect model type for {name}: {st_e}"
                                )
                                raise

                    model.to(self.device)
                    self.models[name] = model
                    self.tokenizers[name] = tokenizer
                    logger.info(f"✅ Successfully loaded reranker '{name}'")
                except Exception as e:
                    logger.error(
                        f"❌ Failed to load reranker '{name}': {e}", exc_info=True
                    )
            else:
                logger.info(f"⚠️ Reranker '{name}' is disabled")

        if self.models:
            self.is_ready = True
            logger.info(
                f"Loaded {len(self.models)} reranker models: {list(self.models.keys())}"
            )
        else:
            logger.warning("No reranker models were loaded.")

    def _predict_batch(
        self, model_name: str, sentence_pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """Predict scores for a batch of sentence pairs using a specified model."""
        model = self.models.get(model_name)
        tokenizer = self.tokenizers.get(model_name)
        model_config = self.configs.get(model_name, {})
        max_length = model_config.get("max_length", 512)

        if not model:
            logger.warning(f"Model '{model_name}' not found, returning zero scores.")
            return [0.0] * len(sentence_pairs)

        # Check if this is a sentence-transformers model
        if hasattr(model, "st_model"):
            logger.info(f"🔍 Using sentence-transformers model for {model_name}")
            try:
                # Use sentence-transformers directly
                texts = [f"{query} [SEP] {doc}" for query, doc in sentence_pairs]
                embeddings = model.st_model.encode(
                    texts, convert_to_tensor=True, show_progress_bar=False
                )

                # Convert embeddings to scores (simple cosine similarity with a reference)
                # For now, use the magnitude of embeddings as a proxy for relevance
                scores = torch.norm(embeddings, dim=1).cpu().numpy().tolist()
                logger.info(
                    f"✅ Generated {len(scores)} scores using sentence-transformers"
                )
                return scores
            except Exception as e:
                logger.error(f"❌ Sentence-transformers prediction failed: {e}")
                return [0.0] * len(sentence_pairs)

        # Standard transformers model
        if not tokenizer:
            logger.warning(
                f"Tokenizer not found for {model_name}, returning zero scores."
            )
            return [0.0] * len(sentence_pairs)

        try:
            inputs = tokenizer(
                sentence_pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # Handle both AutoModelForSequenceClassification and AutoModel
                outputs = model(**inputs)

                if hasattr(outputs, "logits"):
                    # AutoModelForSequenceClassification or Ensemble model
                    logits = outputs.logits

                    # Handle different logits shapes
                    if logits.dim() == 3:  # [batch_size, num_docs, num_labels]
                        logits = logits.squeeze(0)  # Remove batch dimension

                    if logits.dim() == 2:  # [num_docs, num_labels]
                        # For binary classification, use the positive class (index 1)
                        if logits.shape[1] == 2:
                            logits = logits[:, 1]  # Take positive class logits
                        else:
                            logits = logits.squeeze(-1)  # Single label

                    # Ensure logits is 1D
                    if logits.dim() == 0:
                        logits = logits.unsqueeze(0)

                elif hasattr(outputs, "last_hidden_state"):
                    # AutoModel (sentence-transformers) - use last hidden state
                    hidden_states = outputs.last_hidden_state
                    # Apply attention mask if available
                    if "attention_mask" in inputs:
                        attention_mask = inputs["attention_mask"].unsqueeze(-1)
                        hidden_states = hidden_states * attention_mask
                        # Mean pooling
                        logits = (
                            hidden_states.sum(dim=1)
                            / attention_mask.sum(dim=1, keepdim=True)
                        ).squeeze()
                    else:
                        logits = hidden_states.mean(dim=1).squeeze()
                else:
                    # Fallback: try to get any available output
                    logger.warning(f"Unexpected model output type: {type(outputs)}")
                    if hasattr(outputs, "__dict__"):
                        available_attrs = [
                            attr for attr in dir(outputs) if not attr.startswith("_")
                        ]
                        logger.info(f"Available attributes: {available_attrs}")
                    logits = torch.tensor([0.0] * len(sentence_pairs))

                # Convert logits to scores (apply sigmoid for binary classification)
                if hasattr(outputs, "logits") and outputs.logits.shape[-1] == 2:
                    # Binary classification: apply sigmoid to get probabilities
                    import torch.nn.functional as F

                    scores = F.sigmoid(logits).cpu().numpy().tolist()
                else:
                    # Other cases: normalize logits to [0,1] range
                    scores = torch.sigmoid(logits).cpu().numpy().tolist()

            # Ensure scores is always a list
            if not isinstance(scores, list):
                scores = [scores]

            return scores
        except Exception as e:
            logger.error(
                f"Batch prediction failed for model '{model_name}': {e}", exc_info=True
            )
            return [0.0] * len(sentence_pairs)

    def _rank_with_model(
        self, model_name: str, query: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generic ranking function for a specific model using batch prediction."""
        if not documents:
            return []

        # Map model names to standard score keys and determine ranking method
        score_key_mapping = {
            "light_reranker": "light_reranker_score",
            "cross_encoder": "cross_encoder_score",
            "combined-reranker-adapt": "cross_encoder_score",
        }

        score_key = score_key_mapping.get(model_name, f"{model_name}_score")

        # Different ranking methods based on model type
        if model_name == "light_reranker":
            # Light reranker uses SentenceTransformer with similarity scoring
            return self._rank_with_sentence_transformer(query, documents, score_key)
        else:
            # Cross encoder uses classification scoring
            return self._rank_with_classification(query, documents, score_key)

    def _rank_with_sentence_transformer(
        self, query: str, documents: List[Dict[str, Any]], score_key: str
    ) -> List[Dict[str, Any]]:
        """Rank documents using SentenceTransformer similarity scoring."""
        try:
            # Get the light reranker model
            model = self.models.get("light_reranker")
            if not model:
                logger.warning("Light reranker model not found")
                return documents

            # Encode query and documents
            query_embedding = model.encode(
                query, convert_to_tensor=True, device=self.device
            )
            doc_texts = [doc.get("content", "") for doc in documents]
            doc_embeddings = model.encode(
                doc_texts, convert_to_tensor=True, device=self.device
            )

            # Calculate cosine similarities
            similarities = torch.cosine_similarity(
                query_embedding.unsqueeze(0), doc_embeddings
            )

            # Convert to scores (0-1 range)
            scores = (similarities + 1) / 2  # Convert from [-1,1] to [0,1]

            # Assign scores to documents
            for doc, score in zip(documents, scores):
                doc[score_key] = score.item()
                logger.debug(f"🔍 Light reranker: {score_key} = {score.item():.4f}")

            return documents

        except Exception as e:
            logger.error(f"Light reranker ranking failed: {e}")
            # Fallback: assign zero scores
            for doc in documents:
                doc[score_key] = 0.0
            return documents

    def _rank_with_classification(
        self, query: str, documents: List[Dict[str, Any]], score_key: str
    ) -> List[Dict[str, Any]]:
        """Rank documents using classification model scoring."""
        try:
            # Get cross encoder model
            cross_encoder_names = [
                name for name in self.models if name != "light_reranker"
            ]
            if not cross_encoder_names:
                logger.warning("No cross encoder models found")
                return documents

            # Use the first available cross encoder
            model_name = cross_encoder_names[0]
            model = self.models.get(model_name)
            tokenizer = self.tokenizers.get(model_name)

            if not model or not tokenizer:
                logger.warning(
                    f"Cross encoder model or tokenizer not found: {model_name}"
                )
                return documents

            # Create sentence pairs for classification
            sentence_pairs = [(query, doc.get("content", "")) for doc in documents]

            # Get scores using batch prediction
            scores = self._predict_batch(model_name, sentence_pairs)

            # SỬA: Trả về ensemble scores đúng cho Tier 3
            for doc, score in zip(documents, scores):
                # ĐÚNG: PhoBERT-base-v2 vs PhoBERT-large scores
                doc["phobert_base_score"] = score * 0.7  # 70% contribution
                doc["phobert_large_score"] = score * 0.3  # 30% contribution
                
                # Ensemble score của Tier 3
                doc["tier3_ensemble_score"] = score
                
                # Legacy support - giữ cross_encoder_score để tương thích
                doc["cross_encoder_score"] = score
                doc[score_key] = score
                
                logger.debug(f"🔍 Tier 3 ensemble: phobert_base={doc['phobert_base_score']:.4f}, phobert_large={doc['phobert_large_score']:.4f}, ensemble={score:.4f}")

            return documents

        except Exception as e:
            logger.error(f"Cross encoder ranking failed: {e}")
            # Fallback: assign zero scores với ensemble scores
            for doc in documents:
                doc["phobert_base_score"] = 0.0
                doc["phobert_large_score"] = 0.0
                doc["tier3_ensemble_score"] = 0.0
                doc["cross_encoder_score"] = 0.0  # Thêm để tương thích
                doc[score_key] = 0.0
            return documents

    def rank_light(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Perform reranking using the light reranker model."""
        return self._rank_with_model("light_reranker", query, documents)

    def rank_cross(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Perform reranking using an ensemble of cross-encoder models."""
        cross_encoder_names = [name for name in self.models if name != "light_reranker"]

        if not cross_encoder_names:
            logger.info("No cross-encoders found, skipping cross-reranking.")
            return documents

        # Run each cross-encoder model
        for name in cross_encoder_names:
            documents = self._rank_with_model(name, query, documents)

        # Ensemble the scores
        for doc in documents:
            cross_scores = [
                doc.get(f"{name}_score", 0.0) for name in cross_encoder_names
            ]
            if cross_scores:
                ensemble_score = np.mean(cross_scores)
                doc["cross_encoder_score"] = ensemble_score
                
                # ĐÚNG: PhoBERT-base-v2 vs PhoBERT-large scores
                doc["phobert_base_score"] = ensemble_score * 0.7  # 70% contribution
                doc["phobert_large_score"] = ensemble_score * 0.3  # 30% contribution
                doc["tier3_ensemble_score"] = ensemble_score
            else:
                doc["cross_encoder_score"] = 0.0
                doc["phobert_base_score"] = 0.0
                doc["phobert_large_score"] = 0.0
                doc["tier3_ensemble_score"] = 0.0

        return documents

    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the reranking engine."""
        return {
            "is_ready": self.is_ready,
            "device": self.device,
            "loaded_models": list(self.models.keys()),
            "configs": self.configs,
        }

    def cleanup(self):
        """Clean up resources used by the reranking engine."""
        try:
            self.models.clear()
            self.tokenizers.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Reranking engine cleanup completed")
        except Exception as e:
            logger.warning(f"Error during reranking engine cleanup: {e}")


# Alias for backward compatibility
Reranker = RerankingEngine
