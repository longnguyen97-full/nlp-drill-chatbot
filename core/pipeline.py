import faiss
import json
import pickle
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import logging
from typing import List

import config
from core.logging_system import get_logger

# Get logger for this module
logger = get_logger(__name__)


class LegalQAPipeline:
    def __init__(self, use_ensemble=True, use_cascaded_reranking=True):
        """
        Khoi tao pipeline bang cach load tat ca cac thanh phan can thiet.
        Support ensemble reranking voi nhieu Cross-Encoder models.
        Support cascaded reranking voi Light Reranker (fast) + Ensemble (strong).
        """
        # Force CPU mode to avoid CUDA issues completely
        self.device = "cpu"
        logger.info(
            f"Using device: {self.device} (forced CPU mode to avoid CUDA issues)"
        )
        self.last_retrieved_results = []  # THEM: De luu ket qua tang 1
        self.use_ensemble = use_ensemble
        self.use_cascaded_reranking = use_cascaded_reranking

        try:
            logger.info("Loading models and resources...")
            # Tang 1: Retrieval - Force CPU loading
            logger.info("Loading Bi-Encoder on CPU...")
            self.bi_encoder = SentenceTransformer(
                str(config.BI_ENCODER_PATH), device="cpu"
            )
            logger.info("Bi-Encoder loaded successfully on CPU")
            self.faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))
            with open(config.INDEX_TO_AID_PATH, "r", encoding="utf-8") as f:
                self.index_to_aid = json.load(f)

            # Tang 2: Light Re-ranking (Light Reranker) - for cascaded reranking
            if self.use_cascaded_reranking:
                self._load_light_reranker_model()

            # Tang 3: Strong Re-ranking (Ensemble)
            self.cross_encoders = []
            self.cross_encoder_tokenizers = []

            if self.use_ensemble:
                # Load multiple Cross-Encoder models for ensemble
                self._load_ensemble_models()
            else:
                # Load single Cross-Encoder model
                self._load_single_model()

            # Du lieu
            with open(config.AID_MAP_PATH, "rb") as f:
                self.aid_map = pickle.load(f)

            logger.info("Pipeline loaded successfully!")
            self.is_ready = True
        except Exception as e:
            logger.error(f"Critical error initializing pipeline: {e}")
            logger.error(
                "Please ensure models and index files are trained and placed correctly."
            )
            self.is_ready = False

    def _load_ensemble_models(self):
        """Load multiple Cross-Encoder models for ensemble reranking"""
        logger.info("[ENSEMBLE] Loading ensemble Cross-Encoder models...")

        # Model 1: PhoBERT-Law (if available) or base PhoBERT
        if config.PHOBERT_LAW_PATH.exists():
            logger.info("[ENSEMBLE] Loading PhoBERT-Law model...")
            model1_path = str(config.PHOBERT_LAW_PATH)
        else:
            logger.info("[ENSEMBLE] Loading base PhoBERT model...")
            model1_path = str(config.CROSS_ENCODER_PATH)

        tokenizer1 = AutoTokenizer.from_pretrained(model1_path)
        model1 = AutoModelForSequenceClassification.from_pretrained(
            model1_path, device_map="cpu"
        )
        model1.eval()

        self.cross_encoder_tokenizers.append(tokenizer1)
        self.cross_encoders.append(model1)

        # Model 2: XLM-RoBERTa (alternative model for ensemble)
        try:
            logger.info("[ENSEMBLE] Loading XLM-RoBERTa model...")
            model2_path = "xlm-roberta-large"
            tokenizer2 = AutoTokenizer.from_pretrained(model2_path)
            model2 = AutoModelForSequenceClassification.from_pretrained(
                model2_path, num_labels=2, device_map="cpu"
            )
            model2.eval()

            self.cross_encoder_tokenizers.append(tokenizer2)
            self.cross_encoders.append(model2)

            logger.info(
                "[ENSEMBLE] Successfully loaded 2 Cross-Encoder models for ensemble"
            )

        except Exception as e:
            logger.warning(f"[ENSEMBLE] Could not load XLM-RoBERTa model: {e}")
            logger.info("[ENSEMBLE] Using single Cross-Encoder model")

    def _load_light_reranker_model(self):
        """Load Light Reranker model for light reranking in cascaded pipeline"""
        logger.info("[CASCADED] Loading Light Reranker model for light reranking...")

        try:
            if config.LIGHT_RERANKER_PATH.exists():
                logger.info("[CASCADED] Loading trained Light Reranker model...")
                model_path = str(config.LIGHT_RERANKER_PATH)
            else:
                logger.info("[CASCADED] Loading pre-trained Light Reranker model...")
                model_path = config.LIGHT_RERANKER_MODEL_NAME

            self.light_reranker_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.light_reranker_model = (
                AutoModelForSequenceClassification.from_pretrained(
                    model_path, num_labels=2, device_map="cpu"
                )
            )
            self.light_reranker_model.eval()

            logger.info("[CASCADED] Light Reranker model loaded successfully!")

        except Exception as e:
            logger.warning(f"[CASCADED] Could not load Light Reranker model: {e}")
            logger.info("[CASCADED] Cascaded reranking will use only strong models")
            self.light_reranker_model = None
            self.light_reranker_tokenizer = None

    def _load_single_model(self):
        """Load single Cross-Encoder model"""
        logger.info("[SINGLE] Loading single Cross-Encoder model...")

        # Use PhoBERT-Law if available, otherwise use base model
        if config.PHOBERT_LAW_PATH.exists():
            model_path = str(config.PHOBERT_LAW_PATH)
            logger.info("[SINGLE] Using PhoBERT-Law model")
        else:
            model_path = str(config.CROSS_ENCODER_PATH)
            logger.info("[SINGLE] Using base Cross-Encoder model")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, device_map="cpu"
        )
        model.eval()

        self.cross_encoder_tokenizers.append(tokenizer)
        self.cross_encoders.append(model)

    def retrieve_batch(self, queries: List[str], top_k: int):
        """
        TOI UU: Thuc hien Tang 1 (Retrieval) cho mot batch cac cau hoi.
        """
        if not self.is_ready:
            logger.error("Pipeline (Tang 1) chua san sang de truy xuat.")
            return [], []

        logger.info(
            f"Tang 1: Dang truy xuat batch {top_k} ung vien cho {len(queries)} cau hoi..."
        )
        # Ma hoa tat ca cau hoi trong mot batch
        query_embeddings = self.bi_encoder.encode(
            queries, convert_to_tensor=True, device=self.device, show_progress_bar=True
        )
        query_embeddings_np = query_embeddings.cpu().numpy()
        faiss.normalize_L2(query_embeddings_np)

        # Tim kiem tren FAISS cho tat ca cac cau hoi trong mot batch
        distances_batch, indices_batch = self.faiss_index.search(
            query_embeddings_np, top_k
        )

        # Chuyen doi indices thanh AIDs
        results_aids = [
            [self.index_to_aid[i] for i in indices] for indices in indices_batch
        ]

        return results_aids, distances_batch

    def rerank_batch(self, queries: List[str], retrieved_aids_batch: List[List[str]]):
        """
        TOI UU: Thuc hien Tang 2 (Re-ranking) cho mot batch cac cau hoi, voi ho tro ensemble va chunking.
        DA TOI UU: Memory management va error handling.
        """
        if not self.is_ready:
            logger.error("Pipeline (Tang 2) chua san sang de xep hang.")
            return [[] for _ in queries]

        logger.info(
            f"Tang 2: Dang xep hang lai batch {len(queries)} cau hoi voi ensemble..."
        )

        chunk_size = config.CROSS_ENCODER_MAX_LENGTH
        overlap = 50

        cross_encoder_inputs = []  # List cac cap [query, chunk_text]
        chunk_info_map = []  # Map tu index cua chunk ve (chi so cau hoi, aid goc)

        # TOI UU: Memory management - xu ly tung query mot
        for query_idx, query in enumerate(queries):
            query_inputs = []
            query_chunk_info = []

            for aid in retrieved_aids_batch[query_idx]:
                if aid in self.aid_map:
                    passage = self.aid_map[aid]
                    # Use first tokenizer for chunking
                    passage_tokens = self.cross_encoder_tokenizers[0].encode(
                        passage, add_special_tokens=False
                    )

                    if len(passage_tokens) <= chunk_size:
                        query_inputs.append([query, passage])
                        query_chunk_info.append({"query_idx": query_idx, "aid": aid})
                    else:
                        for i in range(0, len(passage_tokens), chunk_size - overlap):
                            chunk_token_ids = passage_tokens[i : i + chunk_size]
                            chunk_text = self.cross_encoder_tokenizers[0].decode(
                                chunk_token_ids
                            )
                            query_inputs.append([query, chunk_text])
                            query_chunk_info.append(
                                {"query_idx": query_idx, "aid": aid}
                            )

            cross_encoder_inputs.extend(query_inputs)
            chunk_info_map.extend(query_chunk_info)

        if not cross_encoder_inputs:
            return [[] for _ in queries]

        # TOI UU: Batch processing voi ensemble
        all_ensemble_scores = []
        batch_size = config.CROSS_ENCODER_BATCH_SIZE

        with torch.no_grad():
            for i in range(0, len(cross_encoder_inputs), batch_size):
                batch_inputs = cross_encoder_inputs[i : i + batch_size]

                batch_scores = []

                # Get scores from each model in ensemble
                for model_idx, (tokenizer, model) in enumerate(
                    zip(self.cross_encoder_tokenizers, self.cross_encoders)
                ):
                    try:
                        # OPTIMIZED TOKENIZATION: Use return_tensors=None for better compatibility
                        tokenized = tokenizer(
                            batch_inputs,
                            padding=True,
                            truncation=True,
                            max_length=chunk_size,
                            return_tensors=None,  # Changed from "pt" to None for better compatibility
                        )

                        # Convert to tensors manually for better control
                        for key in tokenized:
                            if isinstance(tokenized[key], list):
                                tokenized[key] = torch.tensor(tokenized[key])

                        # Move to device with error handling
                        try:
                            tokenized = {
                                k: v.to(self.device) for k, v in tokenized.items()
                            }
                        except RuntimeError as e:
                            if "CUDA" in str(e):
                                logger.warning(
                                    f"CUDA error in batch {i//config.CROSS_ENCODER_BATCH_SIZE}, using CPU"
                                )
                                tokenized = {
                                    k: v.to("cpu") for k, v in tokenized.items()
                                }
                            else:
                                raise e

                        logits = model(**tokenized).logits
                        scores = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
                        batch_scores.append(scores)

                        # TOI UU: Clear memory
                        del tokenized, logits
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except Exception as e:
                        logger.error(
                            f"Loi khi xu ly batch {i//config.CROSS_ENCODER_BATCH_SIZE} voi model {model_idx}: {e}"
                        )
                        # Them scores mac dinh cho batch nay
                        batch_scores.append([0.0] * len(batch_inputs))

                # Combine scores from ensemble models (average)
                if batch_scores:
                    ensemble_scores = []
                    for score_idx in range(len(batch_scores[0])):
                        model_scores = [
                            scores[score_idx]
                            for scores in batch_scores
                            if score_idx < len(scores)
                        ]
                        if model_scores:
                            ensemble_score = sum(model_scores) / len(model_scores)
                            ensemble_scores.append(ensemble_score)
                        else:
                            ensemble_scores.append(0.0)

                    all_ensemble_scores.extend(ensemble_scores)
                else:
                    # Fallback: add default scores
                    all_ensemble_scores.extend([0.0] * len(batch_inputs))

        # TOI UU: Efficient score aggregation
        passage_scores = {}  # Key: (query_idx, aid), Value: max_score
        for i, score in enumerate(all_ensemble_scores):
            info = chunk_info_map[i]
            key = (info["query_idx"], info["aid"])
            if key not in passage_scores or score > passage_scores[key]:
                passage_scores[key] = score

        query_to_results = {i: [] for i in range(len(queries))}
        for (query_idx, aid), score in passage_scores.items():
            query_to_results[query_idx].append((aid, score))

        reranker_aids_batch = []
        for i in range(len(queries)):
            results = sorted(query_to_results[i], key=lambda x: x[1], reverse=True)
            reranker_aids_batch.append([aid for aid, score in results])

        return reranker_aids_batch

    def retrieve(self, query: str, top_k: int):
        """Chi thuc hien Tang 1: Retrieval."""
        if not self.is_ready:
            logger.error("Pipeline (Tang 1) chua san sang de truy xuat.")
            return [], []

        logger.debug(
            f"Tang 1: Dang truy xuat {top_k} ung vien cho query: '{query[:50]}...'"
        )

        # Simple CPU encoding - no device conflicts
        try:
            query_embedding = self.bi_encoder.encode(
                query, convert_to_tensor=True, device="cpu"
            )
        except Exception as e:
            logger.error(f"Error in retrieval encoding: {e}")
            # Try without device specification as last resort
            query_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)

        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
        faiss.normalize_L2(query_embedding_np)

        distances, indices = self.faiss_index.search(query_embedding_np, top_k)

        retrieved_aids = [self.index_to_aid[i] for i in indices[0]]
        return retrieved_aids, distances[0]

    def rerank_light(
        self,
        query: str,
        retrieved_aids: list,
        retrieved_distances: list,
        top_k_light: int = 50,
    ):
        """
        Tang 2: Light Re-ranking voi Light Reranker (fast, small Cross-Encoder)
        Loc xuong top_k_light ung vien chat luong nhat tu 500 xuong 50.
        """
        if not self.is_ready or self.light_reranker_model is None:
            logger.warning(
                "[CASCADED] Light Reranker not available, skipping light reranking"
            )
            return retrieved_aids[:top_k_light], retrieved_distances[:top_k_light]

        logger.debug(
            f"[CASCADED] Tang 2: Light reranking {len(retrieved_aids)} candidates to {top_k_light} with Light Reranker..."
        )

        # Prepare inputs for Light Reranker
        cross_encoder_inputs = []
        aid_to_index = {}

        for i, aid in enumerate(retrieved_aids):
            if aid in self.aid_map:
                passage = self.aid_map[aid]
                cross_encoder_inputs.append([query, passage])
                aid_to_index[aid] = i

        if not cross_encoder_inputs:
            return retrieved_aids[:top_k_light], retrieved_distances[:top_k_light]

        # Score with Light Reranker
        light_scores = []
        batch_size = config.LIGHT_RERANKER_BATCH_SIZE  # Use config parameter

        with torch.no_grad():
            for i in range(0, len(cross_encoder_inputs), batch_size):
                batch_inputs = cross_encoder_inputs[i : i + batch_size]

                try:
                    # Tokenize with Light Reranker
                    tokenized = self.light_reranker_tokenizer(
                        batch_inputs,
                        padding=True,
                        truncation=True,
                        max_length=config.LIGHT_RERANKER_MAX_LENGTH,  # Use config parameter
                        return_tensors=None,  # Avoid tuple index error
                    )

                    # Convert to tensors manually
                    for key in tokenized:
                        if isinstance(tokenized[key], list):
                            tokenized[key] = torch.tensor(tokenized[key])

                    # Move to device
                    tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

                    # Get scores
                    logits = self.light_reranker_model(**tokenized).logits
                    scores = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
                    light_scores.extend(scores)

                    # Clear memory
                    del tokenized, logits
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(
                        f"[CASCADED] Error in light reranking batch {i//batch_size}: {e}"
                    )
                    # Add default scores for this batch
                    light_scores.extend([0.0] * len(batch_inputs))

        # Combine light scores with retrieval scores
        combined_scores = []
        for i, aid in enumerate(retrieved_aids):
            if aid in aid_to_index:
                light_score = light_scores[aid_to_index[aid]]
                retrieval_score = 1.0 - float(
                    retrieved_distances[i]
                )  # Convert distance to score
                # Weighted combination using config parameters
                combined_score = (
                    config.LIGHT_RERANKING_WEIGHT * light_score
                    + config.RETRIEVAL_SCORE_WEIGHT * retrieval_score
                )
                combined_scores.append((aid, combined_score, retrieved_distances[i]))
            else:
                combined_scores.append((aid, 0.0, retrieved_distances[i]))

        # Sort by combined score and return top_k_light
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        top_aids = [item[0] for item in combined_scores[:top_k_light]]
        top_distances = [item[2] for item in combined_scores[:top_k_light]]

        logger.debug(
            f"[CASCADED] Light reranking completed: {len(top_aids)} candidates selected"
        )
        return top_aids, top_distances

    def rerank(self, query: str, retrieved_aids: list, retrieved_distances: list):
        """
        Chi thuc hien Tang 2: Re-ranking, voi ho tro ensemble va chunking cho cac van ban dai.
        """
        if not self.is_ready:
            logger.error("Pipeline (Tang 2) chua san sang de xep hang.")
            return []

        logger.debug(
            f"Tang 2: Dang xep hang lai {len(retrieved_aids)} ung vien cho query '{query[:50]}...' voi ensemble..."
        )

        chunk_size = config.CROSS_ENCODER_MAX_LENGTH
        overlap = 50  # So luong token goi len nhau giua cac chunk

        cross_encoder_inputs = []  # List cac cap [query, chunk_text]
        chunk_info_map = []  # Map tu index cua chunk ve aid goc
        original_passages = {}  # Map tu aid ve noi dung goc
        retrieval_scores_map = {}  # Map tu aid ve diem retrieval

        # 1. Chia van ban thanh cac chunk
        for i, aid in enumerate(retrieved_aids):
            if aid in self.aid_map:
                passage = self.aid_map[aid]
                original_passages[aid] = passage
                retrieval_scores_map[aid] = float(retrieved_distances[i])

                # Use first tokenizer for chunking
                passage_tokens = self.cross_encoder_tokenizers[0].encode(
                    passage, add_special_tokens=False
                )

                if len(passage_tokens) <= chunk_size:
                    cross_encoder_inputs.append([query, passage])
                    chunk_info_map.append({"aid": aid})
                else:
                    for j in range(0, len(passage_tokens), chunk_size - overlap):
                        chunk_token_ids = passage_tokens[j : j + chunk_size]
                        chunk_text = self.cross_encoder_tokenizers[0].decode(
                            chunk_token_ids
                        )
                        cross_encoder_inputs.append([query, chunk_text])
                        chunk_info_map.append({"aid": aid})

        if not cross_encoder_inputs:
            return []

        # 2. Cham diem cac chunk theo batch voi ensemble
        all_ensemble_scores = []

        with torch.no_grad():
            for i in range(
                0, len(cross_encoder_inputs), config.CROSS_ENCODER_BATCH_SIZE
            ):
                batch_inputs = cross_encoder_inputs[
                    i : i + config.CROSS_ENCODER_BATCH_SIZE
                ]

                batch_scores = []

                # Get scores from each model in ensemble
                for model_idx, (tokenizer, model) in enumerate(
                    zip(self.cross_encoder_tokenizers, self.cross_encoders)
                ):
                    try:
                        # OPTIMIZED TOKENIZATION: Use return_tensors=None for better compatibility
                        tokenized = tokenizer(
                            batch_inputs,
                            padding=True,
                            truncation=True,
                            max_length=chunk_size,
                            return_tensors=None,  # Changed from "pt" to None for better compatibility
                        )

                        # Convert to tensors manually for better control
                        for key in tokenized:
                            if isinstance(tokenized[key], list):
                                tokenized[key] = torch.tensor(tokenized[key])

                        # CPU processing - no device conflicts
                        tokenized = {k: v.to("cpu") for k, v in tokenized.items()}
                        logits = model(**tokenized).logits
                        scores = torch.softmax(logits, dim=1)[:, 1].tolist()

                        batch_scores.append(scores)

                        # Clear memory
                        del tokenized, logits
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except Exception as e:
                        logger.error(
                            f"Error processing batch {i//config.CROSS_ENCODER_BATCH_SIZE} with model {model_idx}: {e}"
                        )
                        # Add default scores for this batch and continue
                        try:
                            batch_scores.append([0.0] * len(batch_inputs))
                        except:
                            # Fallback if even this fails
                            batch_scores.append([0.0])

                # Combine scores from ensemble models (average)
                if batch_scores:
                    ensemble_scores = []
                    for score_idx in range(len(batch_scores[0])):
                        model_scores = [
                            scores[score_idx]
                            for scores in batch_scores
                            if score_idx < len(scores)
                        ]
                        if model_scores:
                            ensemble_score = sum(model_scores) / len(model_scores)
                            ensemble_scores.append(ensemble_score)
                        else:
                            ensemble_scores.append(0.0)

                    all_ensemble_scores.extend(ensemble_scores)
                else:
                    # Fallback: add default scores
                    all_ensemble_scores.extend([0.0] * len(batch_inputs))

        # 3. Tong hop diem: lay diem max cua cac chunk cho moi van ban goc
        passage_scores = {}  # Key: aid, Value: max_score
        for i, score in enumerate(all_ensemble_scores):
            aid = chunk_info_map[i]["aid"]
            if aid not in passage_scores or score > passage_scores[aid]:
                passage_scores[aid] = score

        # 4. Tao ket qua cuoi cung voi deduplication
        results = []
        seen_contents = set()

        for aid, rerank_score in passage_scores.items():
            content = original_passages[aid]

            # Simple deduplication: check if content is too similar
            content_normalized = content.lower().strip()
            is_duplicate = False

            for seen_content in seen_contents:
                # Check if content is very similar (90% similarity)
                if self._calculate_similarity(content_normalized, seen_content) > 0.9:
                    is_duplicate = True
                    break

            if not is_duplicate:
                results.append(
                    {
                        "aid": aid,
                        "content": content,
                        "retrieval_score": retrieval_scores_map[aid],
                        "rerank_score": float(rerank_score),
                    }
                )
                seen_contents.add(content_normalized)

        return sorted(results, key=lambda x: x["rerank_score"], reverse=True)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using simple character-based approach"""
        if not text1 or not text2:
            return 0.0

        # Simple character-based similarity
        set1 = set(text1)
        set2 = set(text2)

        if not set1 or not set2:
            return 0.0

        intersection = set1.intersection(set2)
        union = set1.union(set2)

        return len(intersection) / len(union) if union else 0.0

    def predict(self, query: str, top_k_retrieval: int, top_k_final: int):
        """
        Thuc hien quy trinh Cascaded Reranking: 3 tang de tim cau tra loi.
        Tang 1: Bi-Encoder retrieval (500 candidates)
        Tang 2: Light Reranker light reranking (50 candidates)
        Tang 3: Ensemble strong reranking (5 final results)
        """
        # Tang 1: Retrieval
        retrieved_aids, retrieved_distances = self.retrieve(query, top_k_retrieval)

        if not retrieved_aids:
            return []

        # LUU LAI KET QUA TANG 1 (CHO VIEC DANH GIA)
        self.last_retrieved_results = [
            {"aid": aid, "retrieval_score": float(score)}
            for aid, score in zip(retrieved_aids, retrieved_distances)
        ]

        # Tang 2: Light Reranking (nếu có Cascaded Reranking)
        if self.use_cascaded_reranking and self.light_reranker_model is not None:
            logger.info(
                f"[CASCADED] Tang 2: Light reranking {len(retrieved_aids)} -> {config.TOP_K_LIGHT_RERANKING} candidates"
            )
            light_aids, light_distances = self.rerank_light(
                query,
                retrieved_aids,
                retrieved_distances,
                top_k_light=config.TOP_K_LIGHT_RERANKING,
            )
            retrieved_aids = light_aids
            retrieved_distances = light_distances
        else:
            logger.info(
                "[CASCADED] Skipping light reranking, using original candidates"
            )

        # Tang 3: Strong Reranking (Ensemble)
        reranked_results = self.rerank(query, retrieved_aids, retrieved_distances)

        logger.info(
            f"[CASCADED] Du doan hoan thanh. Tra ve {min(top_k_final, len(reranked_results))} ket qua."
        )
        return reranked_results[:top_k_final]


def main_test():
    """Ham test nhanh pipeline."""
    pipeline = LegalQAPipeline()

    if pipeline.is_ready:
        test_query = "Nguoi lao dong co duoc nghi nhung ngay nao?"
        logger.info(f"\nBat dau truy van voi cau hoi: '{test_query}'")
        final_results = pipeline.predict(
            test_query,
            top_k_retrieval=config.TOP_K_RETRIEVAL,
            top_k_final=config.TOP_K_FINAL,
        )

        print("\n--- KET QUA CUOI CUNG ---")
        for res in final_results:
            print(f"AID: {res['aid']} | Re-rank Score: {res['rerank_score']:.4f}")
            print(f"Content: {res['content'][:300]}...")
            print("-" * 20)


if __name__ == "__main__":
    main_test()
