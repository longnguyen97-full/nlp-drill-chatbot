#!/usr/bin/env python3
"""
Aid Normalization Utilities
===========================

Provides a single source of truth for normalizing and comparing AIDs across
data preparation, indexing, retrieval, and evaluation.

Principles:
- Normalize Unicode to NFC
- Unify separators (various dashes → '-', various slashes → '/')
- Collapse repeated underscores and whitespace
- Uppercase alphanumerics for stability
- Provide ASCII-folded canonical form to avoid variance like 'NĐ-CP' vs 'ND-CP'

Expose helpers to normalize single AID, lists, and sets.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Set


_DASHES = [
    "–",  # EN DASH
    "—",  # EM DASH
    "−",  # MINUS SIGN
    "‒",  # FIGURE DASH
    "―",  # HORIZONTAL BAR
]

_SLASHES = [
    "／",  # FULLWIDTH SOLIDUS
    "⁄",   # FRACTION SLASH
]


def _unicode_nfc(text: str) -> str:
    try:
        return unicodedata.normalize("NFC", text)
    except Exception:
        return text


def _fold_ascii(text: str) -> str:
    """Fold diacritics to ASCII while preserving structure.

    - Normalize to NFD, strip combining marks
    - Map Vietnamese đ/Đ to ASCII D
    - Uppercase for stable comparisons
    """
    try:
        text_nfd = unicodedata.normalize("NFD", text)
        stripped = "".join(ch for ch in text_nfd if unicodedata.category(ch) != "Mn")
        stripped = stripped.replace("đ", "d").replace("Đ", "D")
        return stripped.upper()
    except Exception:
        return text.upper()


def canonicalize_aid_preserve(text: str) -> str:
    """Canonicalize AID while preserving Vietnamese letters (NFC).

    Applies:
    - Unicode NFC
    - Unify separators (dashes, slashes)
    - Trim and collapse whitespace
    - Normalize underscores
    - Uppercase letters (including Vietnamese)
    """
    if not isinstance(text, str):
        text = str(text)

    s = _unicode_nfc(text).strip()
    # Unify separators
    for d in _DASHES:
        s = s.replace(d, "-")
    for sl in _SLASHES:
        s = s.replace(sl, "/")
    # Remove spaces around separators
    s = re.sub(r"\s*([_\-/])\s*", r"\1", s)
    # Collapse underscores
    s = re.sub(r"_+", "_", s)
    # Remove duplicate slashes
    s = re.sub(r"/+", "/", s)
    # Remove stray spaces
    s = re.sub(r"\s+", " ", s).strip()
    # Uppercase letters for stability, preserve Vietnamese by NFC first
    # Note: Python upper() preserves Vietnamese letters in uppercase form
    s = s.upper()
    return s


def canonicalize_aid_ascii(text: str) -> str:
    """Canonical ASCII-folded AID suitable for stable equality checks.

    Steps:
    - First apply preserve canonicalization to unify separators and case
    - Then fold to ASCII and uppercase
    """
    preserved = canonicalize_aid_preserve(text)
    return _fold_ascii(preserved)


def canonicalize_aid_list(aids: Iterable[str]) -> List[str]:
    return [canonicalize_aid_ascii(a) for a in aids]


def canonicalize_aid_set(aids: Iterable[str]) -> Set[str]:
    return set(canonicalize_aid_list(aids))


__all__ = [
    "canonicalize_aid_preserve",
    "canonicalize_aid_ascii",
    "canonicalize_aid_list",
    "canonicalize_aid_set",
]


