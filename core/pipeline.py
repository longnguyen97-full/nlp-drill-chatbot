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
from core.logging_utils import get_logger

# Get logger for this module
logger = get_logger(__name__)


class LegalQAPipeline:
    def __init__(self):
        """
        Khoi tao pipeline bang cach load tat ca cac thanh phan can thiet.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.last_retrieved_results = []  # THEM: De luu ket qua tang 1

        try:
            logger.info("Loading models and resources...")
            # Tang 1: Retrieval
            self.bi_encoder = SentenceTransformer(
                str(config.BI_ENCODER_PATH), device=self.device
            )
            self.faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))
            with open(config.INDEX_TO_AID_PATH, "r", encoding="utf-8") as f:
                self.index_to_aid = json.load(f)

            # Tang 2: Re-ranking
            self.cross_encoder_tokenizer = AutoTokenizer.from_pretrained(
                str(config.CROSS_ENCODER_PATH)
            )
            self.cross_encoder_model = (
                AutoModelForSequenceClassification.from_pretrained(
                    str(config.CROSS_ENCODER_PATH)
                ).to(self.device)
            )
            self.cross_encoder_model.eval()

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
        TOI UU: Thuc hien Tang 2 (Re-ranking) cho mot batch cac cau hoi, voi ho tro chunking.
        DA TOI UU: Memory management va error handling.
        """
        if not self.is_ready:
            logger.error("Pipeline (Tang 2) chua san sang de xep hang.")
            return [[] for _ in queries]

        logger.info(
            f"Tang 2: Dang xep hang lai batch {len(queries)} cau hoi voi chunking..."
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
                    passage_tokens = self.cross_encoder_tokenizer.encode(
                        passage, add_special_tokens=False
                    )

                    if len(passage_tokens) <= chunk_size:
                        query_inputs.append([query, passage])
                        query_chunk_info.append({"query_idx": query_idx, "aid": aid})
                    else:
                        for i in range(0, len(passage_tokens), chunk_size - overlap):
                            chunk_token_ids = passage_tokens[i : i + chunk_size]
                            chunk_text = self.cross_encoder_tokenizer.decode(
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

        # TOI UU: Batch processing voi memory management
        all_scores = []
        batch_size = config.CROSS_ENCODER_BATCH_SIZE

        with torch.no_grad():
            for i in range(0, len(cross_encoder_inputs), batch_size):
                batch_inputs = cross_encoder_inputs[i : i + batch_size]

                try:
                    tokenized = self.cross_encoder_tokenizer(
                        batch_inputs,
                        padding=True,
                        truncation=True,
                        max_length=chunk_size,
                        return_tensors="pt",
                    ).to(self.device)

                    logits = self.cross_encoder_model(**tokenized).logits
                    scores = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
                    all_scores.extend(scores)

                    # TOI UU: Clear memory
                    del tokenized, logits
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(f"Loi khi xu ly batch {i//batch_size}: {e}")
                    # Them scores mac dinh cho batch nay
                    all_scores.extend([0.0] * len(batch_inputs))

        # TOI UU: Efficient score aggregation
        passage_scores = {}  # Key: (query_idx, aid), Value: max_score
        for i, score in enumerate(all_scores):
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
        query_embedding = self.bi_encoder.encode(
            query, convert_to_tensor=True, device=self.device
        )
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
        faiss.normalize_L2(query_embedding_np)

        # self.faiss_index.nprobe = 16
        distances, indices = self.faiss_index.search(query_embedding_np, top_k)

        retrieved_aids = [self.index_to_aid[i] for i in indices[0]]
        return retrieved_aids, distances[0]

    def rerank(self, query: str, retrieved_aids: list, retrieved_distances: list):
        """
        Chi thuc hien Tang 2: Re-ranking, voi ho tro chunking cho cac van ban dai.
        """
        if not self.is_ready:
            logger.error("Pipeline (Tang 2) chua san sang de xep hang.")
            return []

        logger.debug(
            f"Tang 2: Dang xep hang lai {len(retrieved_aids)} ung vien cho query '{query[:50]}...' voi chunking..."
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

                passage_tokens = self.cross_encoder_tokenizer.encode(
                    passage, add_special_tokens=False
                )

                if len(passage_tokens) <= chunk_size:
                    cross_encoder_inputs.append([query, passage])
                    chunk_info_map.append({"aid": aid})
                else:
                    for j in range(0, len(passage_tokens), chunk_size - overlap):
                        chunk_token_ids = passage_tokens[j : j + chunk_size]
                        chunk_text = self.cross_encoder_tokenizer.decode(
                            chunk_token_ids
                        )
                        cross_encoder_inputs.append([query, chunk_text])
                        chunk_info_map.append({"aid": aid})

        if not cross_encoder_inputs:
            return []

        # 2. Cham diem cac chunk theo batch
        all_scores = []
        with torch.no_grad():
            for i in range(
                0, len(cross_encoder_inputs), config.CROSS_ENCODER_BATCH_SIZE
            ):
                batch_inputs = cross_encoder_inputs[
                    i : i + config.CROSS_ENCODER_BATCH_SIZE
                ]
                tokenized = self.cross_encoder_tokenizer(
                    batch_inputs,
                    padding=True,
                    truncation=True,
                    max_length=chunk_size,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.cross_encoder_model(**tokenized).logits
                scores = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
                all_scores.extend(scores)

        # 3. Tong hop diem: lay diem max cua cac chunk cho moi van ban goc
        passage_scores = {}  # Key: aid, Value: max_score
        for i, score in enumerate(all_scores):
            aid = chunk_info_map[i]["aid"]
            if aid not in passage_scores or score > passage_scores[aid]:
                passage_scores[aid] = score

        # 4. Tao ket qua cuoi cung
        results = [
            {
                "aid": aid,
                "content": original_passages[aid],
                "retrieval_score": retrieval_scores_map[aid],
                "rerank_score": float(rerank_score),
            }
            for aid, rerank_score in passage_scores.items()
        ]

        return sorted(results, key=lambda x: x["rerank_score"], reverse=True)

    def predict(self, query: str, top_k_retrieval: int, top_k_final: int):
        """
        Thuc hien quy trinh 2 tang de tim cau tra loi.
        """
        retrieved_aids, retrieved_distances = self.retrieve(query, top_k_retrieval)

        if not retrieved_aids:
            return []

        reranked_results = self.rerank(query, retrieved_aids, retrieved_distances)

        # LUU LAI KET QUA TANG 1 (CHO VIEC DANH GIA)
        self.last_retrieved_results = [
            {"aid": aid, "retrieval_score": float(score)}
            for aid, score in zip(retrieved_aids, retrieved_distances)
        ]

        logger.info(
            f"Du doan hoan thanh. Tra ve {min(top_k_final, len(reranked_results))} ket qua."
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
