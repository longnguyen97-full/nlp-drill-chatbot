import pickle
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging
import sys
import os
import torch

# --- Fix import path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from core import setup_logging

setup_logging()


def build_faiss_index_optimized(
    model_path, aid_map_path, index_output_path, map_output_path
):
    """
    OPTIMIZED: Encode entire legal corpus and build FAISS index more efficiently.
    - Automatically use GPU for encoding if available.
    - Use IndexIVFPQ to increase search speed and save memory.
    """
    logging.info("Starting FAISS index building (optimized mode)...")

    # --- Tu dong phat hien thiet bi ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Su dung thiet bi '{device.upper()}' cho viec ma hoa (encoding).")

    try:
        logging.info(f"Loading Bi-Encoder model from: {model_path}")
        # Tai model len thiet bi da chon
        model = SentenceTransformer(str(model_path), device=device)
        with open(aid_map_path, "rb") as f:
            aid_map = pickle.load(f)
    except Exception as e:
        logging.error(f"Loi khi tai model hoac aid_map: {e}", exc_info=True)
        logging.error(
            "Please make sure you have trained the Bi-Encoder and saved it to the correct path."
        )
        return

    aids_in_order = list(aid_map.keys())
    articles_in_order = [aid_map[aid] for aid in aids_in_order]

    logging.info(f"Starting to encode {len(articles_in_order)} articles...")
    corpus_embeddings = model.encode(
        articles_in_order,
        batch_size=config.BI_ENCODER_BATCH_SIZE * 4,  # Tang batch size neu co GPU
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    logging.info("Chuan hoa L2 cac vector (buoc quan trong cho tim kiem IP)...")
    faiss.normalize_L2(corpus_embeddings)

    logging.info("Encoding completed. Starting FAISS index building...")
    embedding_dim = corpus_embeddings.shape[1]

    # --- TOI UU: Su dung IndexIVFPQ cho hieu nang cao hon ---
    # IndexIVFPQ divides vector space into cells (nlist) and uses PQ compression
    # de tang toc do va giam bo nho.
    # nlist = (
    #     128  # So luong cell, nen la mot gia tri trong khoang sqrt(N) voi N la so vector
    # )
    # quantizer = faiss.IndexFlatIP(embedding_dim)  # Index co so
    # index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, 8, 8)

    # logging.info("Starting index training...")
    # index.train(corpus_embeddings)
    # logging.info("Index training completed.")

    index = faiss.IndexFlatIP(embedding_dim)

    logging.info("Them cac vector vao index...")
    index.add(corpus_embeddings)
    logging.info(f"Index built with total {index.ntotal} vectors.")

    # --- Luu ket qua ---
    faiss.write_index(index, str(index_output_path))
    with open(map_output_path, "w", encoding="utf-8") as f:
        json.dump(aids_in_order, f)

    logging.info(f"[SUCCESS] Completed!")
    logging.info(f"Index saved to '{index_output_path}'")
    logging.info(f"Mapping file saved to '{map_output_path}'.")


def main():
    build_faiss_index_optimized(
        model_path=config.BI_ENCODER_PATH,
        aid_map_path=config.AID_MAP_PATH,
        index_output_path=config.FAISS_INDEX_PATH,
        map_output_path=config.INDEX_TO_AID_PATH,
    )


if __name__ == "__main__":
    main()
