"""
Script 00a: Filter dataset to remove samples with inappropriate ground truth
Run BEFORE all other steps to ensure data quality.

Usage:
    python scripts/00a_filter_dataset.py
"""

import json
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Set
from tqdm.auto import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import config
from core.logging_utils import get_logger

# Sử dụng logger đã được setup từ pipeline chính
logger = get_logger(__name__)


class DatasetFilter:
    """
    Filter dataset to remove samples with ground truth that doesn't match the question
    """

    def __init__(self):
        self.legal_terms_mapping = {
            "hai quan": [
                "hai quan",
                "nhap canh",
                "xuat canh",
                "khai bao",
                "bien gioi",
                "cua khau",
            ],
            "ngoai te": ["ngoai te", "usd", "dollar", "tien mat", "ngoai hoi", "do la"],
            "lao dong": [
                "lao dong",
                "nguoi lao dong",
                "nghi phep",
                "lam viec",
                "nhan vien",
                "cong nhan",
                "thoi gian lam viec",
            ],
            "dan su": [
                "dan su",
                "hop dong",
                "boi thuong",
                "tai san",
                "quyen so huu",
                "nghia vu",
            ],
            "hinh su": ["hinh su", "toi pham", "an phat", "tu", "vi pham", "truy cuu"],
            "hon nhan": [
                "hon nhan",
                "gia dinh",
                "ket hon",
                "ly hon",
                "vo chong",
                "con cai",
            ],
            "thua ke": [
                "thua ke",
                "di san",
                "di chuc",
                "de lai",
                "nguoi thua ke",
                "tai san thua ke",
            ],
            "doanh nghiep": [
                "doanh nghiep",
                "cong ty",
                "kinh doanh",
                "dau tu",
                "san xuat",
            ],
            "thue": [
                "thue",
                "khai thue",
                "nop thue",
                "thue thu nhap",
                "thue gia tri gia tang",
            ],
            "bao hiem": [
                "bao hiem",
                "bhxh",
                "bhyt",
                "bao hiem xa hoi",
                "bao hiem y te",
            ],
            "giao thong": [
                "giao thong",
                "lai xe",
                "phuong tien",
                "duong bo",
                "xe co",
                "bang lai xe",
            ],
            "xay dung": ["xay dung", "cong trinh", "kien truc", "nha o", "quy hoach"],
            "moi truong": [
                "moi truong",
                "o nhiem",
                "bao ve",
                "tai nguyen",
                "sinh thai",
            ],
            "giao duc": ["giao duc", "hoc sinh", "sinh vien", "truong hoc", "dao tao"],
            "y te": [
                "y te",
                "benh vien",
                "thuoc",
                "kham chua benh",
                "bac si",
                "dieu tri",
            ],
            "dat dai": ["dat dai", "quyen su dung dat", "chuyen nhuong", "thue dat"],
            "tin dung": ["tin dung", "ngan hang", "vay", "no", "lai suat", "the chap"],
            "bat dong san": ["bat dong san", "nha dat", "can ho", "mua ban nha"],
            "chung khoan": [
                "chung khoan",
                "co phieu",
                "dau tu",
                "thi truong tai chinh",
            ],
            "hanh chinh": [
                "hanh chinh",
                "thu tuc",
                "giay to",
                "cong chuc",
                "quy trinh",
            ],
        }

    def load_data(self):
        """Load all necessary data"""
        logger.info("Loading data...")

        # Load training data
        with open(config.TRAIN_JSON_PATH, "r", encoding="utf-8") as f:
            self.train_data = json.load(f)

        # Load legal corpus
        with open(config.LEGAL_CORPUS_PATH, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        # Tao mapping doc_id -> content
        self.doc_content_map = {}
        for doc in tqdm(self.corpus, desc="Building doc content map"):
            doc_id = doc.get("id")
            content = doc.get("content", [])

            # Extract all article content
            all_content = []
            for article in content:
                article_content = article.get("content_Article", "")
                all_content.append(article_content)

            self.doc_content_map[doc_id] = " ".join(all_content).lower()

        logger.info(f"Loaded {len(self.train_data)} training samples")
        logger.info(f"Loaded {len(self.corpus)} legal documents")

    def extract_question_keywords(self, question: str) -> List[str]:
        """Extract keywords from question"""
        question_lower = question.lower()
        question_keywords = []

        for category, terms in self.legal_terms_mapping.items():
            if any(term in question_lower for term in terms):
                question_keywords.extend(terms)

        return list(set(question_keywords))  # Remove duplicates

    def is_question_relevant_to_doc(self, question: str, doc_id: int) -> bool:
        """
        Check if question is relevant to document
        """
        question_keywords = self.extract_question_keywords(question)

        if not question_keywords:
            # If no keywords found, consider as relevant (conservative approach)
            return True

        doc_content = self.doc_content_map.get(doc_id, "")

        if not doc_content:
            return False

        # Calculate relevance score
        relevance_score = sum(
            1 for keyword in question_keywords if keyword in doc_content
        )

        # Threshold: at least 1 keyword must match
        return relevance_score > 0

    def filter_training_data(self) -> List[Dict]:
        """
        Filter training data to remove samples with inappropriate ground truth
        """
        logger.info("Filtering training data...")

        filtered_data = []
        filtered_out_count = 0

        for sample in tqdm(self.train_data, desc="Filtering training data"):
            question = sample.get("question", "")
            relevant_laws = sample.get("relevant_laws", [])

            if not question or not relevant_laws:
                filtered_out_count += 1
                continue

            # Check if all doc_ids exist in corpus AND at least 1 is relevant
            valid_doc_ids = []
            for doc_id in relevant_laws:
                if doc_id in self.doc_content_map:
                    valid_doc_ids.append(doc_id)

            if not valid_doc_ids:
                # All doc_ids don't exist in corpus
                filtered_out_count += 1
                logger.debug(
                    f"Filtered out (no valid doc_ids): '{question[:60]}...' -> docs {relevant_laws}"
                )
                continue

            # Check if at least 1 valid doc_id is relevant
            is_relevant = False
            for doc_id in valid_doc_ids:
                if self.is_question_relevant_to_doc(question, doc_id):
                    is_relevant = True
                    break

            if is_relevant:
                # Update sample to only include valid doc_ids
                sample["relevant_laws"] = valid_doc_ids
                filtered_data.append(sample)
            else:
                filtered_out_count += 1
                logging.debug(
                    f"Filtered out (not relevant): '{question[:60]}...' -> docs {valid_doc_ids}"
                )

        logging.info(f"Original training data: {len(self.train_data)} samples")
        logging.info(f"Filtered training data: {len(filtered_data)} samples")
        logging.info(
            f"Filtered out: {filtered_out_count} samples ({filtered_out_count/len(self.train_data)*100:.1f}%)"
        )

        return filtered_data

    def analyze_filtering_results(
        self, original_data: List[Dict], filtered_data: List[Dict]
    ):
        """
        Phan tich ket qua filtering
        """
        logging.info("Analyzing filtering results...")

        original_count = len(original_data)
        filtered_count = len(filtered_data)
        removed_count = original_count - filtered_count

        print(f"\n{'='*60}")
        print("[PROGRESS] DATASET FILTERING RESULTS")
        print(f"{'='*60}")
        print(f"[FILE] Original data:      {original_count:,} samples")
        print(f"[OK] Clean data:     {filtered_count:,} samples")
        print(
            f"[FAIL] Removed:      {removed_count:,} samples ({removed_count/original_count*100:.1f}%)"
        )
        print(f"{'='*60}")

        # Phan tich theo domain
        domain_stats = {}
        for sample in filtered_data:
            question = sample.get("question", "").lower()

            for domain, terms in self.legal_terms_mapping.items():
                if any(term in question for term in terms):
                    domain_stats[domain] = domain_stats.get(domain, 0) + 1
                    break

        print(f"\n[NOTE] DOMAIN DISTRIBUTION (Top 10):")
        sorted_domains = sorted(domain_stats.items(), key=lambda x: x[1], reverse=True)
        for domain, count in sorted_domains[:10]:
            percentage = count / filtered_count * 100
            print(f"  {domain:15}: {count:4} samples ({percentage:5.1f}%)")

    def save_filtered_data(self, filtered_data: List[Dict]):
        """
        Save filtered data
        """
        # Backup original file
        backup_path = (
            config.TRAIN_JSON_PATH.parent
            / f"{config.TRAIN_JSON_PATH.stem}_original.json"
        )
        if not backup_path.exists():
            logging.info(f"Backing up original file to: {backup_path}")
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(self.train_data, f, ensure_ascii=False, indent=2)

        # Save filtered data
        logging.info(f"Saving filtered data to: {config.TRAIN_JSON_PATH}")
        with open(config.TRAIN_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)

        # Save filtering report
        report_path = config.DATA_RAW_DIR / "filtering_report.json"
        report = {
            "original_count": len(self.train_data),
            "filtered_count": len(filtered_data),
            "removed_count": len(self.train_data) - len(filtered_data),
            "removal_percentage": (len(self.train_data) - len(filtered_data))
            / len(self.train_data)
            * 100,
            "filtering_criteria": "Semantic relevance between question and ground truth documents",
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logging.info(f"Filtering report saved to: {report_path}")

    def run(self):
        """
        Run the complete filtering process
        """
        try:
            # Load data
            self.load_data()

            # Filter training data
            filtered_data = self.filter_training_data()

            # Analyze results
            self.analyze_filtering_results(self.train_data, filtered_data)

            # Save filtered data
            self.save_filtered_data(filtered_data)

            print(f"\n[SUCCESS] Dataset filtering completed successfully!")
            print(f"[FILE] Filtered data saved to: {config.TRAIN_JSON_PATH}")
            print(
                f"[FILE] Original data backed up to: {config.TRAIN_JSON_PATH.parent / f'{config.TRAIN_JSON_PATH.stem}_original.json'}"
            )

            return True

        except Exception as e:
            logging.error(f"Error during dataset filtering: {e}", exc_info=True)
            return False


def main():
    """
    Main function
    """
    logging.info("=" * 60)
    logging.info("STARTING DATASET FILTERING PROCESS")
    logging.info("=" * 60)

    # Kiem tra file ton tai
    if not config.TRAIN_JSON_PATH.exists():
        logging.error(f"Training data not found: {config.TRAIN_JSON_PATH}")
        return False

    if not config.LEGAL_CORPUS_PATH.exists():
        logging.error(f"Legal corpus not found: {config.LEGAL_CORPUS_PATH}")
        return False

    # Chay filtering
    filter_engine = DatasetFilter()
    success = filter_engine.run()

    if success:
        logging.info("[SUCCESS] Dataset filtering completed successfully!")
        logging.info("[START] You can now run the next steps with the filtered data")
    else:
        logging.error("[FAIL] Dataset filtering failed!")

    return success


if __name__ == "__main__":
    main()
