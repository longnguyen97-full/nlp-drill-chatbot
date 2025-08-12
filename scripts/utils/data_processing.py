#!/usr/bin/env python3
"""
Data Processing Utilities for Training - LawBot v8.1
=====================================================

Utilities specifically for processing data during the training pipeline,
such as selecting candidate AIDs and ranking them based on lexical similarity.
"""

import re
import math
from collections import Counter
from typing import List, Dict

from core.services.logging_service import get_logger

logger = get_logger(__name__)

# -------------------------------
# Text normalization & scoring
# -------------------------------

def _normalize_text(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    # Keep unicode letters/numbers and whitespace
    text = re.sub(r"[^\w\s\u00C0-\u1EF9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return text.split()


def _token_overlap_score(query: str, content: str) -> float:
    q_tokens = _normalize_text(query)
    c_tokens = _normalize_text(content)
    if not q_tokens or not c_tokens:
        return 0.0
    q_counts = Counter(q_tokens)
    c_counts = Counter(c_tokens)
    # Soft overlap: sum of min counts normalized by geometric mean
    common = sum(min(q_counts[t], c_counts[t]) for t in q_counts.keys() & c_counts.keys())
    norm = math.sqrt(sum(q_counts.values()) * sum(c_counts.values()))
    return common / norm if norm > 0 else 0.0


def select_top_aids_for_question(
    question: str, candidate_aids: List[str], aid_map: Dict[str, str], top_k: int = 3
) -> List[str]:
    """Select top-k AIDs within a document for a question using token-overlap scoring."""
    scored = []
    for aid in candidate_aids:
        content = aid_map.get(aid)
        if not content:
            continue
        score = _token_overlap_score(question, content)
        scored.append((score, aid))
    if not scored:
        return []
    scored.sort(reverse=True)
    return [aid for _, aid in scored[:top_k]]


def rank_aids_for_question(
    question: str, candidate_aids: List[str], aid_map: Dict[str, str]
) -> List[tuple]:
    """Return list of (aid, score) sorted by lexical soft-overlap descending."""
    ranked = []
    for aid in candidate_aids:
        content = aid_map.get(aid)
        if not content:
            continue
        score = _token_overlap_score(question, content)
        ranked.append((aid, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
