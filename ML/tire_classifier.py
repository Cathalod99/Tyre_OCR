# tire_classifier.py
# Offline brand + model detection using brand list and optional brand_models.json.
from __future__ import annotations
import json
import os
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

BRANDS: List[str] = [
    "APlus","AUSTONE","Alliance","Altenzo","Antares","Apollo","Arivo","Atlas","Autogreen","Avon",
    "BF Goodrich","Barum","Bridgestone","CST","Ceat","Challenger","Chengshan","Continental","Cooper",
    "Davanti","Dayton","Debica","Delinte","Diamondback","Diplomat","Double Star","Dunlop","Duraturn",
    "Dynamo","Event","Evergreen","Falken","Firemax","Firestone","Fortuna","Fortune","Fronway","Fulda",
    "GT Radial","Gislaved","Goodride","Goodyear","Greentrac","Grenlander","Habilead","Haida","Hankook",
    "Ilink","Infinity","Insa Turbo","Invovic","Iris","Kelly","Kenda","Kingboss","Kingstar","Kleber",
    "Kontio","Kormoran","Kpatos","Kumho","Landsail","Lanvigator","Lappi","Lassa","Laufenn","Leao",
    "Linglong","Marshal","Massimo","Matador","Maxtrek","Maxxis","Mazzini","Michelin","Michelin Collection",
    "Milestone","Milever","Minerva","Mirage","Momo","Nankang","Nexen","Nokian","Nordexx","Nordman","Novex",
    "Onyx","Orium","Ovation","Pace","Paxaro","Petlas","Pirelli","Premiorri","Prinx","Profil","Radar",
    "Radburg","Riken","RoadX","Roadhog","Roadstone","Rotalla","Rovelo","Royal Black","Sailun","Sava",
    "Semperit","Sentury","Sonix","Sportiva","Star Performer","Starmaxx","Sumitomo","Sunny","Sunwide",
    "Superia","Taurus","Tigar","Tomket","Torque","Tourador","Toyo","Tracmax","Trazano","Triangle",
    "Tristar","Uniroyal","Viking","Vittos","Voyager","Vredestein","Waterfall","Westlake","Windforce",
    "Winrun","Yartu","Yokohama","Zeetex"
]

MODEL_STOPWORDS = {
    "TOTAL","PERFORMANCE","TREAD","TREADWEAR","TRACTION","TEMPERATURE","RADIAL","TUBELESS",
    "OUTSIDE","EXTRA","LOAD","WARNING","MAX","PRESS","MADE","IN","POLYESTER","POLYAMIDE",
    "STEEL","SIDEWALL","PLY","PLIES","ROTATION","SAFETY","M+S","XL","RIM","RIMS"
}

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

def _load_model_db() -> Dict[str, List[str]]:
    """
    Optional file: /mnt/data/brand_models.json
    Shape:
      {
        "CONTINENTAL": ["PremiumContact", "SportContact", "EcoContact", ...],
        "EVENT": ["POTENTEM", "FUTURUM", ...]
      }
    """
    paths = ["/mnt/data/brand_models.json", "data/brand_models.json"]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                out = {}
                for k, arr in data.items():
                    out[str(k).upper()] = [str(x).strip() for x in (arr or []) if str(x).strip()]
                return out
            except Exception:
                return {}
    return {}

MODELS = _load_model_db()

def best_brand_match(text: str, threshold: float = 0.30) -> Tuple[Optional[str], float]:
    for brand in BRANDS:
        pat = r"\b" + re.escape(brand).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pat, text, re.IGNORECASE):
            return brand, 1.0
    scores = [(brand, _ratio(text, brand)) for brand in BRANDS]
    best = max(scores, key=lambda x: x[1])
    return (best[0], best[1]) if best[1] >= threshold else (None, 0.0)

def detect_model(text: str, manufacturer: str) -> str:
    """
    Try exact model list first (if available), then heuristic tokens around brand mention.
    """
    if not manufacturer:
        return ""
    brand = manufacturer.upper()

    # 1) Look up in model DB with regex-ish presence in text
    db = MODELS.get(brand, [])
    hits = []
    for m in db:
        if not m:
            continue
        pat = r"\b" + re.escape(m).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pat, text, re.IGNORECASE):
            hits.append(m)
    if hits:
        # prefer the longest / contains digit
        with_digit = [h for h in hits if re.search(r"\d", h)]
        return (with_digit[0] if with_digit else max(hits, key=len))

    # 2) Heuristic: take 50-char window after the brand occurrence
    win = re.search(rf"(?i)\b{re.escape(manufacturer)}\b(.{{0,50}})", text)
    window = win.group(1) if win else text
    tokens = re.findall(r"\b[A-Z][A-Za-z0-9\-]+\b", window.upper())
    tokens = [t for t in tokens if t not in MODEL_STOPWORDS and _norm(t) != _norm(manufacturer)]
    cands = []
    for n in (3, 2, 1):
        for i in range(len(tokens)-n+1):
            phrase = " ".join(tokens[i:i+n])
            score = (1 if re.search(r"\d", phrase) else 0, len(phrase))
            cands.append((score, phrase))
        if cands:
            break
    return (sorted(cands, reverse=True)[0][1] if cands else "").title()