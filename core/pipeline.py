import pickle
import logging
from typing import List, Dict, Any

import config
from core.retrieval import Retriever
from core.reranking import Reranker
from core.services.logging_service import get_logger

logger = get_logger(__name__)

class LegalQAPipeline:
    def __init__(self, use_ensemble=True, use_cascaded_reranking=True):
        self.is_ready = False
        try:
            self.retriever = Retriever()
            self.reranker = Reranker(use_ensemble, use_cascaded_reranking)
            with open(config.AID_MAP_PATH, "rb") as f:
                self.aid_map = pickle.load(f)
            self.is_ready = True
        except Exception as e:
            logger.error(f"Failed to initialize LegalQAPipeline: {e}", exc_info=True)
            raise

    def predict(
        self,
        query: str,
        top_k_retrieval: int,
        top_k_final: int,
        top_k_light_reranking: int = None,
    ) -> List[Dict[str, Any]]:
        # Step 1: Retrieval
        retrieved_aids, retrieved_scores = self.retriever.retrieve(query, top_k_retrieval)

        # Step 2: Reranking
        if self.reranker.use_cascaded_reranking:
             light_aids, light_scores = self.reranker.rerank_light(
                query, retrieved_aids, retrieved_scores, self.aid_map, 
                top_k_light=top_k_light_reranking or config.TOP_K_LIGHT_RERANKING
            )
             retrieved_aids, retrieved_scores = light_aids, light_scores

        reranked_results = self.reranker.rerank(query, retrieved_aids, retrieved_scores, self.aid_map)
        
        return reranked_results[:top_k_final]


def main_test():
    """Ham test nhanh pipeline."""
    pipeline = LegalQAPipeline()

    if pipeline.is_ready:
        test_query = "Nguoi lao dong co duoc nghi nhung ngay nao?"
        logger.info(f"\nBat dau truy van voi cau hoi: '{test_query}'")

        try:
            final_results = pipeline.predict(
                test_query,
                top_k_retrieval=config.TOP_K_RETRIEVAL,
                top_k_final=config.TOP_K_FINAL,
                top_k_light_reranking=config.TOP_K_LIGHT_RERANKING,
            )

            print("\n--- KET QUA CUOI CUNG ---")
            for res in final_results:
                print(f"AID: {res['aid']} | Re-rank Score: {res['rerank_score']:.4f}")
                print(f"Content: {res['content'][:300]}...")
                print("-" * 20)

        except Exception as e:
            logger.error(f"Error in test: {e}")
        finally:
            # The cleanup method was removed from LegalQAPipeline, so this block is removed.
            pass


if __name__ == "__main__":
    main_test()
