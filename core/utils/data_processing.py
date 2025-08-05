#!/usr/bin/env python3
"""
Data Processing Utilities - LawBot v8.0
=======================================

Data loading, parsing, and processing utilities.
"""

import json
import logging
import pickle
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
from core.logging_system import get_logger

logger = get_logger(__name__)


def parse_legal_corpus(corpus_path: Path) -> List[Dict[str, str]]:
    """
    Đọc và xử lý file legal_corpus.json, trả về một danh sách các điều luật.
    Mỗi điều luật là một dict chứa 'aid' và 'content'.

    Args:
        corpus_path: Đường dẫn đến file legal_corpus.json

    Returns:
        List[Dict[str, str]]: Danh sách các điều luật với 'aid' và 'content'
    """
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            legal_corpus_data = json.load(f)

        all_articles = []
        processed_count = 0
        error_count = 0

        for doc in legal_corpus_data:
            try:
                law_id = doc.get("law_id", "unknown")
                content_items = doc.get("content", [])

                if not isinstance(content_items, list):
                    logger.warning(
                        f"[PARSE] Document has non-list content: {type(content_items)}"
                    )
                    error_count += 1
                    continue

                for article in content_items:
                    try:
                        if isinstance(article, dict) and "content_Article" in article:
                            content_text = article.get("content_Article", "").strip()
                            article_id = article.get("aid", "unknown")

                            if content_text:
                                all_articles.append(
                                    {
                                        "aid": f"{law_id}_{article_id}",
                                        "content": content_text,
                                    }
                                )
                                processed_count += 1
                            else:
                                logger.warning(
                                    f"[PARSE] Empty content for article {article_id}"
                                )
                                error_count += 1
                        else:
                            logger.warning(
                                f"[PARSE] Invalid article structure: {type(article)}"
                            )
                            error_count += 1

                    except Exception as e:
                        logger.warning(f"[PARSE] Error processing article: {e}")
                        error_count += 1
                        continue

            except Exception as e:
                logger.warning(f"[PARSE] Error processing document: {e}")
                error_count += 1
                continue

        logger.info(
            f"[PARSE] Successfully parsed {processed_count} articles, {error_count} errors"
        )
        return all_articles

    except Exception as e:
        logger.error(f"[PARSE] Error reading legal corpus: {e}")
        return []


def validate_legal_corpus_structure(corpus_path: Path) -> bool:
    """
    Kiểm tra cấu trúc của legal corpus để đảm bảo tính hợp lệ.

    Args:
        corpus_path: Đường dẫn đến file legal_corpus.json

    Returns:
        bool: True nếu cấu trúc hợp lệ, False nếu không
    """
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.error("[VALIDATE] Legal corpus should be a list of documents")
            return False

        for i, doc in enumerate(data):
            if not isinstance(doc, dict):
                logger.error(f"[VALIDATE] Document {i} should be a dictionary")
                return False

            if "law_id" not in doc:
                logger.error(f"[VALIDATE] Document {i} missing 'law_id'")
                return False

            if "content" not in doc:
                logger.error(f"[VALIDATE] Document {i} missing 'content'")
                return False

            if not isinstance(doc["content"], list):
                logger.error(f"[VALIDATE] Document {i} content should be a list")
                return False

        logger.info("[VALIDATE] Legal corpus structure validation passed")
        return True

    except Exception as e:
        logger.error(f"[VALIDATE] Error validating legal corpus structure: {e}")
        return False


def load_json(file_path: Union[str, Path]) -> Optional[Any]:
    """Load JSON data from file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        return None


def load_pickle(file_path: Union[str, Path]) -> Optional[Any]:
    """Load pickle data from file"""
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load pickle from {file_path}: {e}")
        return None


def load_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """Load JSONL data from file"""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    except Exception as e:
        logger.error(f"Failed to load JSONL from {file_path}: {e}")
        return []


def save_json(data: Any, file_path: Union[str, Path]) -> bool:
    """Save data to JSON file"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        return False


def save_pickle(data: Any, file_path: Union[str, Path]) -> bool:
    """Save data to pickle file"""
    try:
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        logger.error(f"Failed to save pickle to {file_path}: {e}")
        return False


def save_lines(lines: List[str], file_path: Union[str, Path]) -> bool:
    """Save lines to text file"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        return True
    except Exception as e:
        logger.error(f"Failed to save lines to {file_path}: {e}")
        return False


def split_json_data(
    data: List[Dict], train_ratio: float = 0.8, random_seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Split JSON data into train and validation sets"""
    random.seed(random_seed)
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def split_lines_data(
    lines: List[str], train_ratio: float = 0.8, random_seed: int = 42
) -> Tuple[List[str], List[str]]:
    """Split lines data into train and validation sets"""
    random.seed(random_seed)
    random.shuffle(lines)
    split_idx = int(len(lines) * train_ratio)
    return lines[:split_idx], lines[split_idx:]
