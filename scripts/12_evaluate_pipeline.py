#!/usr/bin/env python3
"""
Evaluate Pipeline - Script Danh Gia Pipeline Toi Uu
===================================================

Script nay danh gia toan bo pipeline voi du lieu da duoc fix mapping
va dua ra metrics chinh xac cho evaluation.

Tac gia: LawBot Team
Phien ban: Fixed Evaluation v2.0
"""

import json
import logging
import pickle
import torch
import faiss
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class OptimizedPipelineEvaluator:
    """Evaluator toi uu cho pipeline voi mapping da fix"""

    def __init__(self, k_values=None):
        self.k_values = k_values or [1, 3, 5, 10]
        self.pipeline = None
        self.evaluation_data = None
        self.logger = logging.getLogger(__name__)

    def load_pipeline(self):
        """Load pipeline voi models da train"""
        logger.info("Loading pipeline components...")

        try:
            # Load Bi-Encoder
            from sentence_transformers import SentenceTransformer

            self.bi_encoder = SentenceTransformer(str(config.BI_ENCODER_PATH))

            # Load FAISS index
            self.faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))

            # Load index to AID mapping
            with open(config.INDEX_TO_AID_PATH, "r", encoding="utf-8") as f:
                self.index_to_aid = json.load(f)

            # Load Cross-Encoder (optional - may not be trained yet)
            try:
                from transformers import (
                    AutoTokenizer,
                    AutoModelForSequenceClassification,
                )

                self.cross_encoder_tokenizer = AutoTokenizer.from_pretrained(
                    str(config.CROSS_ENCODER_PATH)
                )
                self.cross_encoder_model = (
                    AutoModelForSequenceClassification.from_pretrained(
                        str(config.CROSS_ENCODER_PATH)
                    )
                )
                self.cross_encoder_model.eval()
                self.cross_encoder_available = True
                logger.info("Cross-Encoder loaded successfully!")
            except Exception as e:
                logger.warning(f"Cross-Encoder not available: {e}")
                logger.warning("Will only evaluate retrieval stage")
                self.cross_encoder_available = False

            # Load AID map
            with open(config.AID_MAP_PATH, "rb") as f:
                self.aid_map = pickle.load(f)

            logger.info("Pipeline components loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
            return False

    def load_evaluation_data(self):
        """Load evaluation data voi format da fix"""
        logger.info("Loading evaluation data...")

        try:
            # Load evaluation data
            evaluation_data_path = config.DATA_PROCESSED_DIR / "evaluation_data.json"
            with open(evaluation_data_path, "r", encoding="utf-8") as f:
                self.evaluation_data = json.load(f)

            logger.info(f"Loaded {len(self.evaluation_data)} evaluation samples")
            return True

        except Exception as e:
            logger.error(f"Error loading evaluation data: {e}")
            return False

    def evaluate_retrieval(
        self, query: str, ground_truth_aids: List[str], top_k: int = 100
    ):
        """Danh gia retrieval stage"""
        # Encode query
        query_embedding = self.bi_encoder.encode([query], convert_to_tensor=True)
        query_embedding_np = query_embedding.cpu().numpy()
        faiss.normalize_L2(query_embedding_np)

        # Search
        distances, indices = self.faiss_index.search(query_embedding_np, top_k)

        # Convert to AIDs
        retrieved_aids = [self.index_to_aid[i] for i in indices[0]]

        return retrieved_aids, distances[0]

    def evaluate_reranking(
        self, query: str, retrieved_aids: List[str], ground_truth_aids: List[str]
    ):
        """Danh gia reranking stage"""
        if not retrieved_aids:
            return []

        # Check if Cross-Encoder is available
        if (
            not hasattr(self, "cross_encoder_available")
            or not self.cross_encoder_available
        ):
            logger.warning("Cross-Encoder not available, returning original ranking")
            return retrieved_aids

        # Prepare inputs for Cross-Encoder
        inputs = []
        for aid in retrieved_aids:
            if aid in self.aid_map:
                content = self.aid_map[aid]
                inputs.append([query, content])

        if not inputs:
            return []

        # Score with Cross-Encoder
        scores = []
        batch_size = 4

        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i : i + batch_size]

                # Tokenize
                tokenized = self.cross_encoder_tokenizer(
                    batch_inputs,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )

                # Get scores
                outputs = self.cross_encoder_model(**tokenized)
                batch_scores = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().tolist()
                scores.extend(batch_scores)

        # Sort by scores
        scored_aids = list(zip(retrieved_aids, scores))
        scored_aids.sort(key=lambda x: x[1], reverse=True)

        return [aid for aid, score in scored_aids]

    def calculate_metrics(self, ground_truth: List[str], retrieved: List[str], k: int):
        """Tinh toan metrics cho mot query"""
        if not retrieved or k == 0:
            return 0.0, 0.0, 0.0

        ground_truth_set = set(ground_truth)
        retrieved_k = retrieved[:k]

        # Precision@k
        relevant_retrieved = sum(1 for item in retrieved_k if item in ground_truth_set)
        precision = relevant_retrieved / len(retrieved_k)

        # Recall@k
        recall = relevant_retrieved / len(ground_truth_set) if ground_truth_set else 0.0

        # F1@k
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return precision, recall, f1

    def run_comprehensive_evaluation(self):
        """Chay danh gia toan dien"""
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE PIPELINE EVALUATION")
        logger.info("=" * 60)

        # Load components
        if not self.load_pipeline():
            return False

        if not self.load_evaluation_data():
            return False

        # Initialize metrics
        retrieval_metrics = {}
        reranking_metrics = {}
        for k in self.k_values:
            retrieval_metrics[f"precision@{k}"] = []
            retrieval_metrics[f"recall@{k}"] = []
            retrieval_metrics[f"f1@{k}"] = []
            reranking_metrics[f"precision@{k}"] = []
            reranking_metrics[f"recall@{k}"] = []
            reranking_metrics[f"f1@{k}"] = []

        # Evaluate each query
        logger.info(f"Evaluating {len(self.evaluation_data)} queries...")

        for i, sample in enumerate(self.evaluation_data):
            query = sample["question"]
            ground_truth_aids = sample["relevant_aids"]

            logger.info(
                f"Evaluating query {i+1}/{len(self.evaluation_data)}: {query[:50]}..."
            )

            # Retrieval evaluation
            retrieved_aids, _ = self.evaluate_retrieval(query, ground_truth_aids)

            # Reranking evaluation
            reranked_aids = self.evaluate_reranking(
                query, retrieved_aids, ground_truth_aids
            )

            # Calculate metrics for each k
            for k in self.k_values:
                # Retrieval metrics
                p, r, f1 = self.calculate_metrics(ground_truth_aids, retrieved_aids, k)
                retrieval_metrics[f"precision@{k}"].append(p)
                retrieval_metrics[f"recall@{k}"].append(r)
                retrieval_metrics[f"f1@{k}"].append(f1)

                # Reranking metrics
                p, r, f1 = self.calculate_metrics(ground_truth_aids, reranked_aids, k)
                reranking_metrics[f"precision@{k}"].append(p)
                reranking_metrics[f"recall@{k}"].append(r)
                reranking_metrics[f"f1@{k}"].append(f1)

        # Calculate averages
        final_retrieval_metrics = {}
        final_reranking_metrics = {}

        for k in self.k_values:
            final_retrieval_metrics[f"precision@{k}"] = np.mean(
                retrieval_metrics[f"precision@{k}"]
            )
            final_retrieval_metrics[f"recall@{k}"] = np.mean(
                retrieval_metrics[f"recall@{k}"]
            )
            final_retrieval_metrics[f"f1@{k}"] = np.mean(retrieval_metrics[f"f1@{k}"])

            final_reranking_metrics[f"precision@{k}"] = np.mean(
                reranking_metrics[f"precision@{k}"]
            )
            final_reranking_metrics[f"recall@{k}"] = np.mean(
                reranking_metrics[f"recall@{k}"]
            )
            final_reranking_metrics[f"f1@{k}"] = np.mean(reranking_metrics[f"f1@{k}"])

        # Display results
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)

        logger.info("RETRIEVAL METRICS:")
        for k in self.k_values:
            p = final_retrieval_metrics[f"precision@{k}"]
            r = final_retrieval_metrics[f"recall@{k}"]
            f1 = final_retrieval_metrics[f"f1@{k}"]
            logger.info(f"  P@{k}: {p:.4f}, R@{k}: {r:.4f}, F1@{k}: {f1:.4f}")

        logger.info("RERANKING METRICS:")
        for k in self.k_values:
            p = final_reranking_metrics[f"precision@{k}"]
            r = final_reranking_metrics[f"recall@{k}"]
            f1 = final_reranking_metrics[f"f1@{k}"]
            logger.info(f"  P@{k}: {p:.4f}, R@{k}: {r:.4f}, F1@{k}: {f1:.4f}")

        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_samples": len(self.evaluation_data),
            "retrieval_metrics": final_retrieval_metrics,
            "reranking_metrics": final_reranking_metrics,
            "mapping_fixed": True,
        }

        # Create reports directory
        config.REPORTS_DIR.mkdir(exist_ok=True)

        # Save detailed results
        report_path = (
            config.REPORTS_DIR
            / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Detailed results saved to: {report_path}")
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

        return True


def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("PIPELINE EVALUATION - FIXED MAPPING VERSION")
    logger.info("=" * 60)

    evaluator = OptimizedPipelineEvaluator()
    success = evaluator.run_comprehensive_evaluation()

    if not success:
        logger.error("Evaluation failed!")
        sys.exit(1)
    else:
        logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
