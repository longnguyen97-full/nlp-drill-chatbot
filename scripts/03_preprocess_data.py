#!/usr/bin/env python3
"""
Preprocess Data - Script Tien Xu Ly Du Lieu Toi Uu
====================================================

Script nay tien xu ly du lieu va tu dong fix mapping issues
de dam bao pipeline hoat dong dung va cho ra metrics evaluation hop ly.

Tac gia: LawBot Team
Phien ban: Fixed Mapping v2.0
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def create_output_dirs():
    """Tao thu muc output neu chua co"""
    dirs = [
        config.DATA_PROCESSED_DIR,
        config.DATA_VALIDATION_DIR,
        config.MODELS_DIR,
        config.INDEXES_DIR,
        config.REPORTS_DIR,
        Path("logs"),
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    logger.info("Created output directories")


def build_maps_optimized(corpus_path, aid_map_output_path, doc_id_map_output_path):
    """Build maps voi fix mapping issues"""
    logger.info("Building optimized maps with mapping fixes...")

    # Load legal corpus
    with open(corpus_path, "r", encoding="utf-8") as f:
        legal_corpus = json.load(f)

    # Load training data
    with open(config.TRAIN_JSON_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # Build aid_map (AID -> content)
    aid_map = {}
    doc_id_to_aids = {}

    logger.info(f"Processing {len(legal_corpus)} legal documents...")

    for doc in legal_corpus:
        doc_id = doc["id"]
        law_id = doc["law_id"]
        content_items = doc.get("content", [])

        # Convert doc_id to string for consistency
        doc_id_str = str(doc_id)

        if doc_id_str not in doc_id_to_aids:
            doc_id_to_aids[doc_id_str] = []

        for item in content_items:
            article_id = item["aid"]
            content = item["content_Article"]

            # Create AID format: law_id_article_id
            aid = f"{law_id}_{article_id}"

            # Store in aid_map
            aid_map[aid] = content

            # Store mapping from doc_id to AIDs
            doc_id_to_aids[doc_id_str].append(aid)

    # Fix training data mapping issues
    logger.info("Fixing training data mapping issues...")
    fixed_train_data = []

    for sample in train_data:
        doc_ids = sample["relevant_laws"]
        valid_aids = []

        # Convert doc_ids to AIDs
        for doc_id in doc_ids:
            doc_id_str = str(doc_id)
            if doc_id_str in doc_id_to_aids:
                aids = doc_id_to_aids[doc_id_str]
                valid_aids.extend(aids)

        if valid_aids:
            # Create fixed sample
            fixed_sample = {
                "qid": sample["qid"],
                "question": sample["question"],
                "relevant_aids": valid_aids,  # Changed from relevant_laws
            }
            fixed_train_data.append(fixed_sample)

    # Save fixed training data
    train_fixed_path = config.DATA_RAW_DIR / "train_fixed.json"
    with open(train_fixed_path, "w", encoding="utf-8") as f:
        json.dump(fixed_train_data, f, ensure_ascii=False, indent=2)

    # Save maps
    with open(aid_map_output_path, "wb") as f:
        pickle.dump(aid_map, f)

    with open(doc_id_map_output_path, "w", encoding="utf-8") as f:
        json.dump(doc_id_to_aids, f, ensure_ascii=False, indent=2)

    # Create evaluation mapping info
    evaluation_mapping = {
        "train_samples": len(fixed_train_data),
        "total_aids": len(aid_map),
        "total_docs": len(doc_id_to_aids),
        "format": "aid_based",
        "mapping_fixed": True,
    }

    evaluation_mapping_path = config.DATA_PROCESSED_DIR / "evaluation_mapping.json"
    with open(evaluation_mapping_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_mapping, f, ensure_ascii=False, indent=2)

    logger.info(f"Created {len(aid_map)} AIDs")
    logger.info(f"Created {len(doc_id_to_aids)} doc mappings")
    logger.info(f"Fixed {len(fixed_train_data)} training samples")
    logger.info("Mapping issues fixed successfully!")

    return aid_map, doc_id_to_aids


def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("PREPROCESS DATA - FIXED MAPPING VERSION")
    logger.info("=" * 60)

    try:
        # Create output directories
        create_output_dirs()

        # Build optimized maps with fixes
        aid_map, doc_id_to_aids = build_maps_optimized(
            config.LEGAL_CORPUS_PATH,
            config.AID_MAP_PATH,
            config.DOC_ID_TO_AIDS_PATH,
        )

        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Next steps:")
        logger.info("1. Run data preparation with fixed data")
        logger.info("2. Train models with correct mapping")
        logger.info("3. Evaluate with proper metrics")

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
