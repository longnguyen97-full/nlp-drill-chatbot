import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Any, Tuple

import config
from core.services.logging_service import get_logger

logger = get_logger(__name__)

def _load_model(model_path: str, tokenizer_path: str, model_class, tokenizer_class, **kwargs):
    """A general-purpose model and tokenizer loader."""
    try:
        tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        model = model_class.from_pretrained(model_path, **kwargs)
        model.eval()
        logger.info(f"Successfully loaded model from {model_path}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}", exc_info=True)
        raise

class Reranker:
    def __init__(self, use_ensemble=True, use_cascaded_reranking=True):
        self.device = "cpu"
        self.use_ensemble = use_ensemble
        self.use_cascaded_reranking = use_cascaded_reranking
        
        # Light Reranker
        if self.use_cascaded_reranking:
            self.light_reranker_model, self.light_reranker_tokenizer = _load_model(
                str(config.LIGHT_RERANKER_PATH), str(config.LIGHT_RERANKER_PATH),
                AutoModelForSequenceClassification, AutoTokenizer, num_labels=2, device_map=self.device
            )

        # Strong Reranker
        self.cross_encoders = []
        self.cross_encoder_tokenizers = []
        if self.use_ensemble:
            model1_path = str(config.CROSS_ENCODER_PATH)
            if config.PHOBERT_LAW_PATH.exists():
                model1_path = str(config.PHOBERT_LAW_PATH)
            
            model1, tokenizer1 = _load_model(
                model1_path, model1_path, AutoModelForSequenceClassification, AutoTokenizer, device_map=self.device
            )
            self.cross_encoders.append(model1)
            self.cross_encoder_tokenizers.append(tokenizer1)

            try:
                model2, tokenizer2 = _load_model(
                    "xlm-roberta-large", "xlm-roberta-large", AutoModelForSequenceClassification, AutoTokenizer, num_labels=2, device_map=self.device
                )
                self.cross_encoders.append(model2)
                self.cross_encoder_tokenizers.append(tokenizer2)
            except Exception as e:
                logger.warning(f"Could not load XLM-RoBERTa for ensemble, proceeding with one model. Error: {e}")
        else:
            model, tokenizer = _load_model(
               str(config.CROSS_ENCODER_PATH), str(config.CROSS_ENCODER_PATH), AutoModelForSequenceClassification, AutoTokenizer, device_map=self.device
            )
            self.cross_encoders.append(model)
            self.cross_encoder_tokenizers.append(tokenizer)
    
    def _chunk_passage(self, passage: str, chunk_size: int, overlap: int) -> List[str]:
        tokenizer = self.cross_encoder_tokenizers[0]
        tokens = tokenizer.encode(passage, add_special_tokens=False)
        if not tokens:
            return []
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_token_ids = tokens[i : i + chunk_size]
            chunk_text = tokenizer.decode(chunk_token_ids)
            chunks.append(chunk_text)
        return chunks

    def rerank_light(self, query: str, retrieved_aids: list, retrieved_scores: list, aid_map: dict, top_k_light: int = 50) -> Tuple[List[str], List[float]]:
        if not retrieved_aids:
            return [], []

        pairs = []
        valid_aids = []
        for aid in retrieved_aids:
            content = aid_map.get(aid)
            if content:
                pairs.append([query, content])
                valid_aids.append(aid)

        if not pairs:
            return [], []

        with torch.no_grad():
            inputs = self.light_reranker_tokenizer(
                pairs, padding=True, truncation=True, return_tensors='pt', max_length=config.LIGHT_RERANKER_MAX_LENGTH
            ).to(self.device)
            logits = self.light_reranker_model(**inputs, return_dict=True).logits
            scores = logits.softmax(dim=1)[:, 1].cpu().numpy()

        scored_aids = sorted(zip(valid_aids, scores), key=lambda x: x[1], reverse=True)
        
        top_aids = [aid for aid, score in scored_aids[:top_k_light]]
        top_scores = [float(score) for aid, score in scored_aids[:top_k_light]]
        
        return top_aids, top_scores

    def rerank(self, query: str, retrieved_aids: list, retrieved_scores: list, aid_map: dict) -> List[Dict[str, Any]]:
        if not retrieved_aids:
            return []

        pairs = []
        valid_aids = []
        retrieval_score_map = {aid: score for aid, score in zip(retrieved_aids, retrieved_scores)}

        for aid in retrieved_aids:
            content = aid_map.get(aid)
            if content:
                pairs.append([query, content])
                valid_aids.append(aid)
        
        if not pairs:
            return []

        all_scores = []
        with torch.no_grad():
            for model, tokenizer in zip(self.cross_encoders, self.cross_encoder_tokenizers):
                inputs = tokenizer(
                    pairs, padding=True, truncation=True, return_tensors='pt', max_length=config.CROSS_ENCODER_MAX_LENGTH
                ).to(self.device)
                logits = model(**inputs, return_dict=True).logits
                
                # Assuming classification with 2 labels, take the score for the "relevant" class (index 1)
                scores = logits.softmax(dim=1)[:, 1]
                all_scores.append(scores.cpu().numpy())

        final_scores = np.mean(all_scores, axis=0)
        logger.debug(f"Final aggregated scores shape: {final_scores.shape}")

        results = []
        for i, aid in enumerate(valid_aids):
            try:
                results.append({
                    "aid": aid,
                    "content": aid_map.get(aid, ""),
                    "retrieval_score": float(retrieval_score_map.get(aid, 0.0)),
                    "rerank_score": float(final_scores[i])
                })
            except IndexError:
                logger.error(
                    f"IndexError in rerank: index {i} is out of bounds for final_scores with shape {final_scores.shape}. "
                    f"This likely means the number of valid_aids ({len(valid_aids)}) does not match the number of scores. "
                    f"Problematic AID: {aid}"
                )
                # Skip this problematic entry but continue processing others
                continue

        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        logger.debug(f"Reranking complete. Returning {len(results)} results.")
        
        return results
