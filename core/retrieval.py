#!/usr/bin/env python3
"""
Retrieval Engine - LawBot v8.1 (Refactored)
===========================================

Retrieval component that uses bi-encoder models and a FAISS index to find
relevant legal documents. It now self-manages content lookups.
"""

import json
from typing import List, Dict, Any
import logging
from pathlib import Path

try:
    import torch
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    from core.utils.logging_manager import get_logger
    from core.utils.parent_law_manager import ensure_parent_law_mapping
except ImportError as e:
    logging.error(f"Import error in retrieval.py: {e}")
    raise

logger = get_logger(__name__)


class RetrievalEngine:
    """Retrieval engine using a bi-encoder and FAISS index."""

    def __init__(
        self,
        bi_encoder_path: str,
        faiss_index_path: str,
        content_map_path: str,
        index_to_aid_path: str,
    ):
        """Initializes the retrieval engine."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_ready = False
        logger.info(f"Using device: {self.device}")

        try:
            logger.info(f"Loading bi-encoder from: {bi_encoder_path}")
            self.bi_encoder = SentenceTransformer(bi_encoder_path, device=self.device)

            logger.info(f"Loading FAISS index from: {faiss_index_path}")
            self.faiss_index = faiss.read_index(faiss_index_path)

            logger.info(f"Loading content map from: {content_map_path}")
            with open(content_map_path, "r", encoding="utf-8") as f:
                self.corpus_content = json.load(f)

            logger.info(f"Loading index-to-AID map from: {index_to_aid_path}")
            with open(index_to_aid_path, "r", encoding="utf-8") as f:
                self.index_to_aid = json.load(f)

            # Load AID to parent law mapping
            # Use utility to ensure mapping exists and is valid
            logger.info("Ensuring parent law mapping is available...")
            if ensure_parent_law_mapping():
                aid_to_parent_path = Path("features/aid_to_parent_law.json")
                with open(aid_to_parent_path, "r", encoding="utf-8") as f:
                    self.aid_to_parent_law = json.load(f)
                logger.info(f"Loaded {len(self.aid_to_parent_law)} parent law mappings")
                # Log a few sample mappings for verification
                sample_mappings = list(self.aid_to_parent_law.items())[:3]
                logger.info(f"Sample mappings: {sample_mappings}")
            else:
                logger.warning(
                    "Failed to ensure parent law mapping, parent law names will not be available"
                )
                self.aid_to_parent_law = {}

            self.is_ready = True
            logger.info(
                f"Retrieval engine initialized successfully. Index size: {self.faiss_index.ntotal}"
            )
        except FileNotFoundError as e:
            logger.error(f"A required mapping file was not found: {e}")
            self.is_ready = False
            raise
        except Exception as e:
            logger.error(f"Failed to initialize retrieval engine: {e}", exc_info=True)
            self.is_ready = False
            raise

    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Retrieves relevant documents for a single query."""
        # Reuse the batch method for a single query
        results = self.retrieve_batch([query], top_k=top_k)
        return results[0] if results else []

    def retrieve_batch(
        self, queries: List[str], top_k: int = 100
    ) -> List[List[Dict[str, Any]]]:
        """Retrieves relevant documents for a batch of queries."""
        if not self.is_ready:
            raise RuntimeError("Retrieval engine is not ready.")
        if not queries:
            return []

        try:
            # Encode queries in a batch
            query_embeddings = self.bi_encoder.encode(
                queries, convert_to_tensor=True, device=self.device
            )
            query_embeddings_np = query_embeddings.cpu().numpy()
            faiss.normalize_L2(query_embeddings_np)

            # Search in a batch
            scores_batch, indices_batch = self.faiss_index.search(
                query_embeddings_np, top_k
            )

            batch_results = []
            for i, query in enumerate(queries):
                scores = scores_batch[i]
                indices = indices_batch[i]

                results = []
                for rank, (score, idx) in enumerate(zip(scores, indices)):
                    if idx < 0:  # Invalid index
                        continue

                    aid = self.index_to_aid.get(str(idx))
                    # Convert AID to string for corpus_content lookup
                    aid_str = str(aid) if aid is not None else None
                    content_info = self.corpus_content.get(aid_str)

                    if content_info:
                        # Get parent law name from mapping
                        parent_law_name = self.aid_to_parent_law.get(
                            str(aid), "Unknown"
                        )

                        # Debug logging for parent law lookup
                        if parent_law_name == "Unknown":
                            logger.info(
                                f"Parent law not found for AID {aid} (type: {type(aid)})"
                            )
                            logger.info(
                                f"Available AIDs in mapping: {list(self.aid_to_parent_law.keys())[:5]}"
                            )
                        else:
                            logger.info(
                                f"Found parent law '{parent_law_name}' for AID {aid}"
                            )

                        results.append(
                            {
                                "aid": aid,
                                "parent_law_name": parent_law_name,
                                "content": (
                                    content_info
                                    if isinstance(content_info, str)
                                    else str(content_info)
                                ),
                                "retrieval_score": float(score),
                                "retrieval_rank": rank + 1,
                            }
                        )
                batch_results.append(results)

            logger.debug(f"Retrieved documents for {len(queries)} queries.")
            return batch_results
        except Exception as e:
            logger.error(f"Error in retrieve_batch: {e}", exc_info=True)
            return [[] for _ in queries]

    def get_index_info(self) -> Dict[str, Any]:
        """Gets information about the FAISS index."""
        if not self.is_ready:
            return {"status": "Engine not ready"}
        return {
            "index_size": self.faiss_index.ntotal,
            "index_dimension": self.faiss_index.d,
            "aid_mapping_size": len(self.index_to_aid),
            "content_map_size": len(self.corpus_content),
            "device": self.device,
        }

    def cleanup(self):
        """Cleans up resources used by the retrieval engine."""
        try:
            del self.bi_encoder
            del self.faiss_index
            del self.corpus_content
            del self.index_to_aid
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Retrieval engine cleanup completed")
        except AttributeError:
            pass  # Attributes may not exist if __init__ failed
        except Exception as e:
            logger.warning(f"Error during retrieval engine cleanup: {e}")


# Alias for backward compatibility
Retriever = RetrievalEngine
