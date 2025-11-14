# size_extractor.py
# Offline tire size + LI/speed extraction and normalization.
from __future__ import annotations
import re
from collections import Counter
from typing import Optional, Tuple, Dict

SIZE_PAT = re.compile(
    r"\b(\d{3})\s*/\s*(\d{2})\s*(?:[Zz]\s*)?[Rr]\s*(\d{2}(?:\.\d)?)\b"
)

# Also accept slightly malformed "ZR" like 245/40ZR18
SIZE_PAT_ZR = re.compile(
    r"\b(\d{3})\s*/\s*(\d{2})\s*[Zz][Rr]\s*(\d{2}(?:\.\d)?)\b"
)

# Commercial tire patterns (LT - Light Truck format)
SIZE_PAT_LT = re.compile(
    r"\b(\d{3})\s*/\s*(\d{2})\s*[Ll][Tt]\s*(\d{2}(?:\.\d)?)\b"
)

# Commercial tire patterns (C - Commercial format)
SIZE_PAT_C = re.compile(
    r"\b(\d{3})\s*/\s*(\d{2})\s*[Cc]\s*(\d{2}(?:\.\d)?)\b"
)

def _format_match(groups: Tuple[str, str, str], suffix: str = "R") -> str:
    width, aspect, rim = groups
    if suffix == "LT":
        return f"{width}/{aspect}LT{rim}"
    if suffix == "C":
        return f"{width}/{aspect}C{rim}"
    return f"{width}/{aspect}R{rim}"


def _size_candidates(text: str) -> Dict[str, int]:
    """
    Collect all plausible size strings and count their occurrences.
    """
    counts: Counter[str] = Counter()
    positions: Dict[str, int] = {}

    if not text:
        return {}

    for pattern, suffix in (
        (SIZE_PAT, "R"),
        (SIZE_PAT_ZR, "R"),
        (SIZE_PAT_LT, "LT"),
        (SIZE_PAT_C, "C"),
    ):
        for match in pattern.finditer(text):
            width, aspect, rim = match.group(1), match.group(2), match.group(3)
            candidate = _format_match((width, aspect, rim), suffix)
            counts[candidate] += 1
            positions.setdefault(candidate, match.start())

    if not counts:
        return {}

    return {cand: counts[cand] for cand in counts}


def _score_size(size_str: str, count: int) -> Tuple[float, int]:
    """
    Score a normalized size string. Higher is better.
    """
    score = float(count) * 100.0
    try:
        width = int(size_str[:3])
    except ValueError:
        width = 0
    if 125 <= width <= 445:
        score += 5.0
    if ".5" in size_str:
        score += 3.0
    if width >= 300:
        score += 1.0
    if width == 0:
        score -= 5.0
    return score, width


def extract_size(text: str) -> Optional[str]:
    candidates = _size_candidates(text)
    if not candidates:
        return None

    best = max(
        candidates.items(),
        key=lambda item: (_score_size(item[0], item[1]), -len(item[0])),
    )[0]
    return best

def extract_li_speed(text: str) -> Optional[str]:
    """
    Find a plausible Load Index / Speed symbol near size or anywhere,
    while avoiding units (PSI/KPA/LBS/KG).
    """
    if not text:
        return None
    for m in re.finditer(r"\b(\d{2,3})\s*([A-Z])\b", text):
        li_s, sp = m.group(1), m.group(2).upper()
        try:
            li = int(li_s)
        except ValueError:
            continue
        if not (60 <= li <= 200):  # Extended range for commercial truck tires
            continue
        ctx = text[max(0, m.start()-10): m.end()+10].upper()
        if any(u in ctx for u in ["PSI","KPA","LBS","LB","KG","KGS"]):
            continue
        if sp == "R" and re.match(r"\s*(1[0-9]|2[0-6])\b", text[m.end():m.end()+4]):
            # This is probably part of "R16" from size, not speed
            continue
        return f"{li}/{sp}"
    return None