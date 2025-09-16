# text_processor.py
# Offline orchestrator: ties together size_extractor, tire_classifier, dot_extractor.
from __future__ import annotations
import re
import json
from typing import Any, Dict

from .size_extractor import extract_size, extract_li_speed
from .tire_classifier import best_brand_match, detect_model
from .dot_extractor import parse_tin

def _normalize_size(s: str) -> str:
    if not s:
        return ""
    m = re.match(r"^\s*(\d{3})\s*/\s*(\d{2})\s*[Zz]?[Rr]\s*(\d{2})\s*$", s)
    return f"{m.group(1)}/{m.group(2)}R{m.group(3)}" if m else s.replace(" ", "").upper()

def _normalize_li_speed(s: str) -> str:
    if not s:
        return ""
    m = re.match(r"^\s*(\d{2,3})\s*[/]?\s*([A-Z])\s*$", s)
    return f"{m.group(1)}/{m.group(2)}" if m else s.upper().replace(" ", "")

def build_result(ocr_text: str) -> Dict[str, Any]:
    size = extract_size(ocr_text) or ""
    li_speed = extract_li_speed(ocr_text) or ""

    brand, score = best_brand_match(ocr_text)
    manufacturer = (brand or "").strip()

    model = detect_model(ocr_text, manufacturer).strip()

    tin = parse_tin(ocr_text)
    full_dot = tin["full"].strip()
    parts = tin["parts"]
    date = parts.get("Date", "")
    wk = parts.get("Week", "")
    yr = parts.get("Year", "")

    out = {
        "Manufacturer": manufacturer,
        "Tire model": model,
        "Tire size": _normalize_size(size),
        "Load index and speed rating": _normalize_li_speed(li_speed),
        "Scan TIN": {
            "Full DOT": full_dot,
            "DOT Code": date,
            "Week Code": wk,
            "Year Code": yr,
            "Plant Code": parts.get("Plant", ""),
            "Plant Name": parts.get("PlantName", ""),
            "Plant Country": parts.get("PlantCountry", ""),
            "Plant Manufacturer": parts.get("PlantManufacturer", "")
        }
    }
    return out

if __name__ == "__main__":
    sample = """
    EVENT POTENTEM UHP 205/55R16 94W
    DOT 10U AVEUHP 3223
    MAX LOAD 670KG (1477LBS) MAX PRESS 340KPA (50PSI)
    """
    print(json.dumps(build_result(sample), indent=2))