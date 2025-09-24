# size_extractor.py
# Offline tire size + LI/speed extraction and normalization.
from __future__ import annotations
import re
from typing import Optional, Tuple

SIZE_PAT = re.compile(
    r"\b(\d{3})\s*/\s*(\d{2})\s*(?:[Zz]\s*)?[Rr]\s*(\d{2})\b"
)

# Also accept slightly malformed "ZR" like 245/40ZR18
SIZE_PAT_ZR = re.compile(
    r"\b(\d{3})\s*/\s*(\d{2})\s*[Zz][Rr]\s*(\d{2})\b"
)

# Commercial tire patterns (LT - Light Truck format)
SIZE_PAT_LT = re.compile(
    r"\b(\d{3})\s*/\s*(\d{2})\s*[Ll][Tt]\s*(\d{2})\b"
)

# Commercial tire patterns (C - Commercial format)
SIZE_PAT_C = re.compile(
    r"\b(\d{3})\s*/\s*(\d{2})\s*[Cc]\s*(\d{2})\b"
)

def extract_size(text: str) -> Optional[str]:
    if not text:
        return None
    
    # Standard radial pattern (XXX/YYRZZ)
    m = SIZE_PAT.search(text)
    if m:
        return f"{m.group(1)}/{m.group(2)}R{m.group(3)}"
    
    # ZR pattern (XXX/YYZRZZ)
    mz = SIZE_PAT_ZR.search(text)
    if mz:
        return f"{mz.group(1)}/{mz.group(2)}R{mz.group(3)}"
    
    # Light Truck pattern (XXX/YYLTZZ)
    ml = SIZE_PAT_LT.search(text)
    if ml:
        return f"{ml.group(1)}/{ml.group(2)}LT{ml.group(3)}"
    
    # Commercial pattern (XXX/YYCZZ)
    mc = SIZE_PAT_C.search(text)
    if mc:
        return f"{mc.group(1)}/{mc.group(2)}C{mc.group(3)}"
    
    return None

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