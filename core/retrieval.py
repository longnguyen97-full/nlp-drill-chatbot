import faiss
import json
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple

import config
from core.services.logging_service import get_logger

logger = get_logger(__name__)

class Retriever:
    def __init__(self):
        self.device = "cpu"
        self.bi_encoder = SentenceTransformer(str(config.BI_ENCODER_PATH), device=self.device)
        self.faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        with open(config.INDEX_TO_AID_PATH, "r", encoding="utf-8") as f:
            self.index_to_aid = json.load(f)

    def retrieve(self, query: str, top_k: int) -> Tuple[List[str], np.ndarray]:
        query_embedding = self.bi_encoder.encode(query, convert_to_tensor=True, device=self.device)
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
        faiss.normalize_L2(query_embedding_np)
        scores, indices = self.faiss_index.search(query_embedding_np, top_k)
        retrieved_aids = [self.index_to_aid[i] for i in indices[0]]
        return retrieved_aids, scores[0]

    def retrieve_batch(self, queries: List[str], top_k: int) -> Tuple[List[List[str]], np.ndarray]:
        query_embeddings = self.bi_encoder.encode(
            queries,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False,
        )
        query_embeddings_np = query_embeddings.cpu().numpy()
        faiss.normalize_L2(query_embeddings_np)
        scores_batch, indices_batch = self.faiss_index.search(query_embeddings_np, top_k)
        results_aids = [[self.index_to_aid[i] for i in indices] for indices in indices_batch]
        return results_aids, scores_batch
