# tire_classifier.py
# Robust brand + model extraction for noisy, multi-line tyre OCR.
# - Brand detection: exact → alias → fuzzy (per-word → whole-text)
# - Model detection (date-friendly, line-aware):
#     * dictionary-first via brand_models.json (optional)
#     * line-window around brand (±2 before, +3 after)
#     * DOTALL brand-centered window (±120 chars)
#     * whole-text fallback
#   Scoring: (frequency in full OCR, appears on brand-near lines, safety-line penalty,
#             tech-tag penalty (if other candidates exist), has-digit, length)
# - Strong, explicit stopword set to kill boilerplate terms (e.g., "DUE", "UNDERINFLATION").
# - Never blacklist real models (e.g., TURANZA, POTENZA, PRIMACY, etc.).

from __future__ import annotations
import json
import os
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

# ------------------------------
# Brand catalogue (extend as needed)
# ------------------------------
BRANDS: List[str] = [
    "APlus","AUSTONE","Alliance","Altenzo","Antares","Apollo","Arivo","Atlas","Autogreen","Avon",
    "BF Goodrich","Barum","Bridgestone","CST","Ceat","Challenger","Chengshan","Continental","Cooper",
    "Davanti","Dayton","Debica","Delinte","Diamondback","Diplomat","Double Star","Dunlop","Duraturn",
    "Dynamo","Event","Evergreen","Falken","Farroad","Firemax","Firestone","Fortuna","Fortune","Fronway","Fulda",
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
    "Winrun","Yartu","Yokohama","Zeetex",
    "Michelin Commercial","Bridgestone Commercial","Continental Commercial","Goodyear Commercial",
    "Firestone Commercial","Pirelli Commercial","Dunlop Commercial","Cooper Commercial"
]

# Common printed variants → canonical names
BRAND_ALIASES: Dict[str, str] = {
    "BFGOODRICH": "BF Goodrich",
    "B.F.GOODRICH": "BF Goodrich",
    "BRIDGESTONE": "Bridgestone",
    "MICHELIN": "Michelin",
    "GOODYEAR": "Goodyear",
    "PIRELLI": "Pirelli",
    "CONTINENTAL": "Continental",
}

# Do NOT put real model names here.
MODEL_STOPWORDS = {
    # generic
    "TOTAL","PERFORMANCE","TREAD","TREADWEAR","TRACTION","TEMPERATURE","RADIAL","TUBELESS",
    "OUTSIDE","EXTRA","LOAD","WARNING","MAX","PRESS","PRESSURE","MADE","IN","POLYESTER","POLYAMIDE",
    "STEEL","SIDEWALL","PLY","PLIES","ROTATION","SAFETY","M+S","XL","RIM","RIMS","DOT","OUT","SIDE",
    "ROAD","KPA","PSI","BAR","MAXLOAD","MAXPRESS","TEMPERATUREA","TEMPERATUREB","TRACTIONA","TRACTIONB",
    # boilerplate/safety phrases that pollute models
    "DUE","UNDERINFLATION","OVERLOADING","FOLLOW","OWNER","MANUAL","VEHICLE","EXPLOSION","IMPROPER",
    "MOUNTING","NEVER","EXCEED","SEAT","BEADS","SERIOUS","INJURY","RECOMMENDED","INFLATE","ONLY",
    "SPECIALLY","TRAINED","PERSONS","SHOULD","TIRES","FAILURE","ASSEMBLY","COLD","SINGLE","DUAL",
    "MAX","LOAD","WAY","MAY","PRESS","APO","TL"
}

# Tech tags / side marks that are not usually the commercial model; we *demote* them if we have a better candidate.
TECH_TAGS = {
    "ENLITEN","ECOPIA","ECO","RUNFLAT","MOE","AO","ROF","SSR","MO","NCS","PNCS","CINTURATO","MFS","RSC"
}

# Optional small, built-in fallback model families for big brands
FALLBACK_MODELS: Dict[str, List[str]] = {
    "BRIDGESTONE": ["TURANZA", "POTENZA", "DUELER", "ECOPIA", "BLIZZAK", "ALENZA", "DURAVIS"],
    "MICHELIN":    ["PRIMACY", "PILOT", "X-ICE", "ENERGY", "LATITUDE", "CROSSCLIMATE"],
    "GOODYEAR":    ["EFFICIENTGRIP", "EAGLE", "VECTOR", "ULTRAGRIP"],
    "CONTINENTAL": ["PREMIUMCONTACT", "SPORTCONTACT", "ECOCONTACT", "VANCONTACT", "ALLSEASONCONTACT"],
    "PIRELLI":     ["CINTURATO", "P ZERO", "SCORPION", "WINTER"],
    "DUNLOP":      ["SPORT", "SP SPORT", "WINTER", "STREETRESPONSE"],
    "FARROAD":     ["FRD 66", "FRD66", "FRD-66"],
    # Commercial/Truck tire models
    "MICHELIN COMMERCIAL": ["XDE", "XDE2", "XDE3", "XZE", "XZE2", "XZE3", "XDA", "XDA2", "XDA3"],
    "BRIDGESTONE COMMERCIAL": ["R250", "R268", "R284", "M729", "M729F", "M729A", "R192", "R192F"],
    "GOODYEAR COMMERCIAL": ["G286", "G288", "G292", "G394", "G395", "G397", "G622", "G622R"],
    "CONTINENTAL COMMERCIAL": ["HDL2", "HDL3", "HDL4", "HDL5", "HDL6", "HDL7", "HDL8", "HDL9"],
    "FIRESTONE COMMERCIAL": ["FS560", "FS561", "FS562", "FS563", "FS564", "FS565", "FS566", "FS567"]
}

# ------------------------------
# Helpers
# ------------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

def _load_model_db() -> Dict[str, List[str]]:
    """
    Optional JSON:
      /mnt/data/brand_models.json   (fallback: data/brand_models.json)
      {
        "BRIDGESTONE": ["Turanza", "Potenza", "Dueler", "Ecopia"],
        ...
      }
    """
    paths = ["/mnt/data/brand_models.json", "data/brand_models.json"]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                out: Dict[str, List[str]] = {}
                for k, arr in data.items():
                    out[str(k).upper()] = [str(x).strip() for x in (arr or []) if str(x).strip()]
                return out
            except Exception:
                return {}
    return {}

MODELS = _load_model_db()

def _word_tokens(text: str) -> List[str]:
    return re.findall(r"\b[A-Za-z][A-Za-z0-9\-\+]*\b", text)

def _best_exact_or_alias(text: str) -> Optional[str]:
    up = text.upper()
    # exact brand with flexible whitespace
    for brand in BRANDS:
        pat = r"\b" + re.escape(brand).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pat, up, re.IGNORECASE):
            return brand
    # alias
    for alias, canonical in BRAND_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", up):
            return canonical
    return None

# ------------------------------
# Brand Detection
# ------------------------------
def best_brand_match(text: str, threshold: float = 0.30) -> Tuple[Optional[str], float]:
    """
    Returns (brand, score). Score ∈ [0,1].
    Strategy:
      1) Exact or alias match → (brand, 1.0)
      2) Fuzzy per-word vs brand list (>= threshold)
      3) Fuzzy whole-text fallback (>= threshold)
    """
    exact = _best_exact_or_alias(text)
    if exact:
        return exact, 1.0

    up = text.upper()
    words = [w for w in _word_tokens(up) if len(w) >= 4]
    best_brand: Optional[str] = None
    best_score = 0.0

    for brand in BRANDS:
        for w in words:
            sc = _ratio(w, brand)
            if sc > best_score and sc >= threshold:
                best_score = sc
                best_brand = brand

    if best_brand:
        return best_brand, best_score

    scores = [(brand, _ratio(up, brand)) for brand in BRANDS]
    brand, sc = max(scores, key=lambda x: x[1])
    return (brand, sc) if sc >= threshold else (None, 0.0)

# ------------------------------
# Model Detection
# ------------------------------
def _tokens_model_like(text: str) -> List[str]:
    """
    Extract tokens that look like model names:
      - start with a letter
      - allow digits and dashes
      - length >= 3 (e.g., TURANZA, ENLITEN, X-ICE, PRIMACY4)
    """
    up = text.upper()
    toks = re.findall(r"\b[A-Z][A-Z0-9\-]{2,}\b", up)
    # Filter stopwords and obvious junk
    toks = [t for t in toks if t not in MODEL_STOPWORDS]
    return toks

def _plausible_model_token(token: str) -> bool:
    """
    Heuristic gate to avoid serial numbers / batch codes:
      - must contain at least one alpha char (already true)
      - if length > 5, require at least 2 alpha chars
      - deny tokens that are overwhelmingly numeric
    """
    letters = sum(ch.isalpha() for ch in token)
    digits = sum(ch.isdigit() for ch in token)
    if letters == 0:
        return False
    if len(token) > 5 and letters < 2:
        return False
    if digits >= len(token) - 1:
        # allow short forms like "R250" (len 4), but block long serials
        if not (len(token) <= 4 and letters >= 1):
            return False
    return True

def _brand_line_indices(text: str, manufacturer: str) -> List[int]:
    if not manufacturer:
        return []
    lines = text.splitlines()
    inds = []
    brand_re = re.compile(rf"(?i)\b{re.escape(manufacturer)}\b")
    brand_up_re = re.compile(rf"(?i)\b{re.escape(manufacturer.upper())}\b")
    for i, ln in enumerate(lines):
        if brand_re.search(ln) or brand_up_re.search(ln):
            inds.append(i)
    return inds

def _window_lines(text: str, centers: List[int], before: int = 2, after: int = 3) -> str:
    lines = text.splitlines()
    keep: List[str] = []
    for c in centers:
        lo = max(0, c - before)
        hi = min(len(lines), c + after + 1)
        keep.extend(lines[lo:hi])
    return "\n".join(keep)

def _is_safety_line(line: str) -> bool:
    # if a line contains several boilerplate markers, treat it as a safety line
    markers = (
        "WARNING","EXPLOSION","UNDERINFLATION","OVERLOADING","OWNER","MANUAL","VEHICLE",
        "IMPROPER","MOUNT","NEVER","EXCEED","PSI","KPA","LOAD","INJURY","RECOMMENDED"
    )
    up = line.upper()
    hit = sum(1 for m in markers if m in up)
    return hit >= 2

def detect_model(text: str, manufacturer: str) -> str:
    """
    Model extraction:
      1) brand_models.json (dictionary-first) over the full OCR text.
      2) Use tokens from lines near the brand (±2…+3 lines).
      3) Brand-centered DOTALL window (±120 chars).
      4) Whole-text fallback.
      Scoring tuple: (frequency, near-brand bonus, safety-line penalty, tech-tag penalty, has-digit, length).
    """
    if not manufacturer:
        return ""
    brand = manufacturer.strip()
    brand_up = brand.upper()

    # 1) Dictionary-first across entire OCR (plus fallback families if no JSON)
    dict_list = MODELS.get(brand_up) or FALLBACK_MODELS.get(brand_up, [])
    if dict_list:
        hits = []
        for m in dict_list:
            if not m:
                continue
            pat = r"\b" + re.escape(m).replace(r"\ ", r"\s+") + r"\b"
            if re.search(pat, text, flags=re.IGNORECASE):
                hits.append(m)
        if hits:
            # prefer the longest with digits, else longest
            with_digit = [h for h in hits if re.search(r"\d", h)]
            choice = (max(with_digit, key=len) if with_digit else max(hits, key=len))
            # preserve original casing if possible
            m2 = re.search(rf"\b({re.escape(choice)})\b", text, flags=re.IGNORECASE)
            return m2.group(1).strip() if m2 else choice.title()

    # 2) Line-aware region around brand mentions
    centers = _brand_line_indices(text, brand)
    lines = text.splitlines()
    region = _window_lines(text, centers) if centers else ""
    region_tokens = _tokens_model_like(region) if region else []

    # 3) Brand-centered DOTALL window (±120 chars both sides)
    if not region_tokens:
        m = re.search(rf"(?is)(.{{0,120}})\b{re.escape(brand)}\b(.{{0,120}})", text)
        if not m:
            m = re.search(rf"(?is)(.{{0,120}})\b{re.escape(brand_up)}\b(.{{0,120}})", text)
        if m:
            win = (m.group(1) or "") + " " + brand + " " + (m.group(2) or "")
            region_tokens = _tokens_model_like(win)

    # 4) Whole-text fallback
    if not region_tokens:
        region_tokens = _tokens_model_like(text)

    # Remove the brand token itself (any case)
    region_tokens = [t for t in region_tokens if _norm(t) != _norm(brand_up)]
    if not region_tokens:
        return ""
    region_tokens = [t for t in region_tokens if _plausible_model_token(t)]
    if not region_tokens:
        return ""

    # Build per-line lookup for penalties/bonuses
    full_up = text.upper()
    brand_near_up = region.upper() if region else ""
    safety_lines_idx = {i for i, ln in enumerate(lines) if _is_safety_line(ln)}
    token_in_safety_lines: Dict[str, int] = {}
    token_in_brand_lines: Dict[str, int] = {}

    # map token appearances to line indices
    for i, ln in enumerate(lines):
        up = ln.upper()
        for t in set(region_tokens):
            if re.search(rf"\b{re.escape(t)}\b", up):
                if i in safety_lines_idx:
                    token_in_safety_lines[t] = token_in_safety_lines.get(t, 0) + 1
                if centers and any(abs(i - c) <= 3 for c in centers):
                    token_in_brand_lines[t] = token_in_brand_lines.get(t, 0) + 1

    uniq = sorted(set(region_tokens))
    scored = []
    for t in uniq:
        freq = len(re.findall(rf"\b{re.escape(t)}\b", full_up))
        near = 1 if (brand_near_up and re.search(rf"\b{re.escape(t)}\b", brand_near_up)) else 0
        safety_pen = token_in_safety_lines.get(t, 0)  # higher → worse
        tech_pen = 1 if (t in TECH_TAGS) else 0
        has_digit = 1 if re.search(r"\d", t) else 0
        score = (freq, near, -safety_pen, -tech_pen, has_digit, len(t))
        scored.append((score, t))

    scored.sort(reverse=True)
    best_tok = scored[0][1]

    # If the winner is a tech tag but there exists a non-tech candidate with decent freq, switch to it
    if best_tok in TECH_TAGS:
        non_tech = [t for (_, t) in scored if t not in TECH_TAGS]
        if non_tech:
            best_tok = non_tech[0]

    # Preserve original casing from the OCR if possible
    m2 = re.search(rf"\b({re.escape(best_tok)})\b", text, flags=re.IGNORECASE)
    return m2.group(1).strip() if m2 else best_tok.title()

# ------------------------------
# Convenience
# ------------------------------
def extract_brand_and_model(text: str) -> Tuple[Optional[str], float, str]:
    """
    Returns (brand, brand_score, model).
    """
    brand, score = best_brand_match(text)
    model = detect_model(text, brand or "")
    return brand, score, model