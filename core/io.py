#!/usr/bin/env python3
"""
File I/O and Path Utilities - LawBot v8.1
=========================================

Centralized utilities for file and path management.
"""

import sys
import json
import pickle
from pathlib import Path
from typing import List, Union, Optional, Any, Dict
from core.services.logging_service import get_logger
from core.aid_utils import canonicalize_aid_ascii

logger = get_logger(__name__)


def create_directories(directories: List[Union[str, Path]]) -> None:
    """Create directories if they don't exist"""
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def setup_script_paths():
    """Setup Python path for script imports"""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def load_config():
    """Load configuration module"""
    setup_script_paths()
    try:
        import config
        return config
    except ImportError as e:
        logger.error(f"Failed to import config: {e}")
        return None

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

def save_json(data: Any, file_path: Path, indent: int = 4, lines: bool = False):
    """
    Lưu dữ liệu vào file JSON.

    Args:
        data: Dữ liệu cần lưu.
        file_path: Đường dẫn đến file output.
        indent: Thụt lề cho file JSON.
        lines: Nếu True, lưu mỗi phần tử trong danh sách thành một dòng (JSONL).
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        if lines and isinstance(data, list):
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            json.dump(data, f, ensure_ascii=False, indent=indent)

def save_pickle(data: Any, file_path: Union[str, Path]) -> bool:
    """Save data to pickle file"""
    try:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
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


def parse_legal_corpus(corpus_path: Path) -> List[Dict[str, Any]]:
    """
    Đọc và xử lý file legal_corpus.json, tạo ra một danh sách phẳng các điều luật.
    Mỗi điều luật là một dict chứa 'aid' (ID toàn cục) và 'content'.
    """
    logger.info(f"Parsing legal corpus from: {corpus_path}")
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            legal_data = json.load(f)

        all_articles = []
        for doc in legal_data:
            law_id = doc.get("law_id", "unknown_law")
            articles_in_doc = doc.get("content", [])
            for article in articles_in_doc:
                article_id = article.get("aid")
                content = article.get("content_Article", "").strip()
                
                if article_id is not None and content:
                    # Tạo một ID toàn cục duy nhất
                    global_aid = f"{law_id}_{article_id}"
                    all_articles.append({
                        "aid": global_aid,
                        "content": content
                    })
        
        logger.info(f"Successfully parsed {len(all_articles)} articles from corpus.")
        return all_articles

    except Exception as e:
        logger.error(f"Error reading legal corpus: {e}", exc_info=True)
        return []
