# tire_classifier.py
# Robust brand + model extraction for noisy, multi-line tyre OCR (cars + trucks).
# - Brand: token-voting with Levenshtein + SequenceMatcher, OCR-aware aliases, optional plant-code boost
# - Model: dictionary-first, then brand-near phrases (1–3 grams), demote boilerplate/tech tags/sizes/countries
# - Keeps same public API: best_brand_match, detect_model, extract_brand_and_model

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
    "Winrun","Yartu","Yokohama","Zeetex",
    # common CV/Truck families show up in OCR too
    "Bridgestone Commercial","Continental Commercial","Goodyear Commercial","Michelin Commercial",
    "Firestone Commercial","Pirelli Commercial","Dunlop Commercial","Cooper Commercial"
]

# Normalize obvious printed variants → canonical (strict)
BRAND_ALIASES: Dict[str, str] = {
    "BFGOODRICH": "BF Goodrich",
    "B.F.GOODRICH": "BF Goodrich",
    "BRIDGESTONE": "Bridgestone",
    "MICHELIN": "Michelin",
    "GOODYEAR": "Goodyear",
    "PIRELLI": "Pirelli",
    "CONTINENTAL": "Continental",
}

# OCR-ish alias hints (loose). If these tokens appear as words, map them.
# Helps fix LIRELLI->Pirelli, MRIDGESTONE->Bridgestone, MICHEA->Michelin, etc.
FUZZY_ALIASES: Dict[str, str] = {
    "LIRELLI": "Pirelli",
    "MRIDGESTONE": "Bridgestone",
    "BRIDGESTONF": "Bridgestone",
    "MICHEA": "Michelin",
    "MICHELINO": "Michelin",
    "GOOPYEAR": "Goodyear",
    "YOKOHOMA": "Yokohama",
    "P.LINGLONG": "Linglong",
    "PLINGLONG": "Linglong",
}

# ------------------------------
# Stopwords / tech tags (do NOT include real models)
# ------------------------------
MODEL_STOPWORDS = {
    # generic junk / boilerplate
    "TOTAL","PERFORMANCE","TREAD","TREADWEAR","TRACTION","TEMPERATURE","RADIAL","TUBELESS","REGROOVABLE",
    "OUTSIDE","EXTRA","LOAD","WARNING","MAX","PRESS","PRESSURE","MADE","IN","ONLY","POLYESTER","POLYAMIDE","NYLON",
    "STEEL","SIDEWALL","PLY","PLIES","ROTATION","SAFETY","DOT","OUT","SIDE","KPA","PSI","BAR","MAXLOAD",
    "MAXPRESS","TEMPERATUREA","TEMPERATUREB","TRACTIONA","TRACTIONB","RECOMMENDED","INFLATE","COLD","SINGLE",
    "DUAL","SERIOUS","INJURY","FAILURE","EXPLOSION","MOUNT","MOUNTING","NEVER","EXCEED","SEAT","BEADS",
    "PERSONS","SPECIALLY","TRAINED","OWNER","MANUAL","VEHICLE","FOLLOW","ASSEMBLY","WARNING","SERIE","SERIES",
    "TRAILER","GRAND","TYRE","READY","STORAGE","CARGO","GARAGE","FITMENT","FRT",
    # units and ranges that creep in
    "XL","M+S","RIM","RIMS","PR","LOADRANGE","LOAD-RANGE","LOADINDEX","LI","SL","TL","REINFORCED",
    # truck boilerplate that often wins incorrectly
    "ALL","ALLSTEEL","ALL-STEEL","ALLSTEELRADIAL","ALL-STEEL-RADIAL","STEELRADIAL",
    # countries / origins that OCR turns into "models"
    "CHINA","VIETNAM","VIETNA","SERBIA","SPAIN","KOREA","JAPAN","USA","POLAND","FRANCE","GERMANY",
    "ITALY","ROMANIA","THAILAND","TAIWAN","CZECH","SLOVAKIA","TURKEY","INDIA"
}

# Tech tags / marks we demote if better candidates exist (but still allow if nothing else appears)
TECH_TAGS = {
    "ENLITEN","ECOPIA","ECO","RUNFLAT","MO","AO","ROF","SSR","NCS","PNCS","RSC","MFS","MOE",
    "REGROOVABLE","ALLSTEEL","ALL-STEEL","ALLSTEELRADIAL","TUBELESS"
}

# Patterns that are NOT models (sizes, LI/Speed, homologation codes)
SIZE_PAT = re.compile(r"\b\d{3}/\d{2}R\d{2}[A-Z0-9]?\b")                 # 235/65R16C
LI_SPEED_PAT = re.compile(r"\b\d{2,3}\s*/\s*\d{2,3}\s*[A-Z]?\b")        # 115/113R
HOMO_PAT = re.compile(r"\b(ECE|E4|DOT|T\w+|OE)\b", re.I)

# Optional small, built-in fallback families (helps when no JSON is present)
FALLBACK_MODELS: Dict[str, List[str]] = {
    "BRIDGESTONE": ["TURANZA","POTENZA","DUELER","ECOPIA","BLIZZAK","ALENZA","DURAVIS"],
    "MICHELIN":    ["PRIMACY","PILOT","X-ICE","ENERGY","LATITUDE","CROSSCLIMATE","X MULTI","X LINE","X WORKS"],
    "GOODYEAR":    ["EFFICIENTGRIP","EAGLE","VECTOR","ULTRAGRIP","KMAX","FUELMAX","OMNITRAC","URBANMAX"],
    "CONTINENTAL": ["PREMIUMCONTACT","SPORTCONTACT","ECOCONTACT","VANCONTACT","ALLSEASONCONTACT","HYDRO","HYDROPLUS"],
    "PIRELLI":     ["CINTURATO","P ZERO","SCORPION","WINTER","CARRIER","ITINERIS","ITINERIS T90"],
    "DUNLOP":      ["SPORT","SP SPORT","WINTER","STREETRESPONSE"],
    # Truck lines
    "MICHELIN COMMERCIAL": ["XZE","XDE","XDA","X MULTI","X LINE","X WORKS"],
    "BRIDGESTONE COMMERCIAL": ["R268","R284","M729","R192","DURAVIS"],
    "GOODYEAR COMMERCIAL": ["KMAX","FUELMAX","OMNITRAC","URBANMAX"],
    "CONTINENTAL COMMERCIAL": ["HYDRO","HYDROPLUS","HDL","HSC","HSR","HDR"],
    "FIRESTONE COMMERCIAL": ["FS561","FD690","FT491","TRANSFORCE"]
}

# Normalize multi-word → canonical join for common truck/car families
PHRASE_NORMALISERS: Dict[str, str] = {
    "MULTI ROUTE": "MultiRoute",
    "X MULTI": "X MULTI",
    "ALL SEASON": "AllSeason",
    "CROSS CLIMATE": "CrossClimate",
    "P ZERO": "P ZERO",
    "SP SPORT": "SP SPORT",
    "ITINERIS T90": "ITINERIS T90",
}

# ------------------------------
# Optional plant-code DB (for brand hint)
# ------------------------------
def _load_plant_db() -> Dict[str, Dict[str, str]]:
    paths = [os.environ.get("PLANT_CODES_JSON"), "/mnt/data/plant_codes.json", "data/plant_codes.json"]
    db: Dict[str, Dict[str, str]] = {}
    for p in paths:
        if not p:
            continue
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                if isinstance(data, dict):
                    for code, meta in data.items():
                        db[str(code).upper()] = {k: str(v) for k, v in (meta or {}).items()}
                elif isinstance(data, list):
                    for row in data:
                        code = str(row.get("code","")).upper()
                        if not code: continue
                        db[code] = {
                            "manufacturer": str(row.get("manufacturer","")),
                            "plant": str(row.get("plant","")),
                            "country": str(row.get("country","")),
                        }
                break
            except Exception:
                pass
    return db

PLANTS = _load_plant_db()

def _parse_plant_hint(text: str) -> Optional[str]:
    """
    Very small parser: find 'DOT' then next 1-2 tokens; take 2–3 alnum as plant code,
    map to manufacturer if available.
    """
    # keep raw tokens so we can see brackets/slashes
    tokens = re.findall(r"[A-Za-z0-9\(\)\[\]\-/\.]+", text)
    def _clean(tok: str, digits: bool=False) -> str:
        t = tok.strip()
        t = re.sub(r"^[\(\[\{]+|[\)\]\}]+$", "", t)
        t = re.sub(r"[\/\.\-]+", "", t)
        t = t.upper()
        if digits:
            t = t.translate(str.maketrans({"O":"0","I":"1","L":"1"}))
        return re.sub(r"[^A-Z0-9]","",t)
    for i, tok in enumerate(tokens):
        if _clean(tok) == "DOT":
            for j in range(i+1, min(i+6, len(tokens))):
                c = _clean(tokens[j], digits=True)
                if 2 <= len(c) <= 3 and c.isalnum():
                    meta = PLANTS.get(c)
                    if meta:
                        m = meta.get("manufacturer","").strip()
                        if m:
                            return m
    return None

# ------------------------------
# Utils
# ------------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

def _levenshtein(a: str, b: str) -> int:
    a, b = _norm(a), _norm(b)
    if a == b:
        return 0
    if not a: return len(b)
    if not b: return len(a)
    n, m = len(a), len(b)
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev = dp[0]
        dp[0] = i
        ca = a[i-1]
        for j in range(1, m+1):
            cb = b[j-1]
            cost = 0 if ca == cb else 1
            dp[j], prev = min(dp[j]+1, dp[j-1]+1, prev+cost), dp[j]
    return dp[m]

def _word_tokens(text: str) -> List[str]:
    return re.findall(r"\b[A-Za-z][A-Za-z0-9\.\-\+]*\b", text)

def _strict_exact_or_alias(text: str) -> Optional[str]:
    up = text.upper()
    # exact brand with flexible whitespace
    for brand in BRANDS:
        pat = r"\b" + re.escape(brand).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pat, up, re.IGNORECASE):
            return brand
    # alias (strict)
    for alias, canonical in BRAND_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", up):
            return canonical
    return None

# ------------------------------
# Brand Detection (new voting approach)
# ------------------------------
def best_brand_match(text: str, threshold: float = 0.30) -> Tuple[Optional[str], float]:
    """
    Returns (brand, score in [0,1]).
    Strategy:
      0) strict exact/alias match
      1) token voting with Levenshtein + SequenceMatcher (length & confidence weighted)
      2) full-text fuzzy as tie-breaker
      + plant code manufacturer hint boost, if resolvable
    """
    if not text:
        return (None, 0.0)

    # 0) strict hit wins
    exact = _strict_exact_or_alias(text)
    if exact:
        return exact, 1.0

    # fuzzy aliases / OCR oddities
    up_words = [w.upper() for w in _word_tokens(text)]
    for w in up_words:
        if w in FUZZY_ALIASES:
            return FUZZY_ALIASES[w], 0.95

    # Brand hint from DOT plant code (if any)
    plant_brand_hint = _parse_plant_hint(text)

    # 1) token voting
    votes: Dict[str, float] = {b: 0.0 for b in BRANDS}
    words = [w for w in _word_tokens(text) if len(w) >= 4]
    if not words:
        return (plant_brand_hint, 0.6) if plant_brand_hint else (None, 0.0)

    # per-token: find best brand candidate; weight by token length + confidence
    for w in words:
        w_clean = re.sub(r"[^A-Za-z0-9]", "", w)
        if not w_clean:
            continue
        # fast path: if a fuzzy alias word appears, count it strongly
        if w_clean.upper() in FUZZY_ALIASES:
            brand = FUZZY_ALIASES[w_clean.upper()]
            votes[brand] = votes.get(brand, 0.0) + 2.0
            continue
        best_b, best_s = None, 0.0
        for brand in BRANDS:
            # compute two signals and take max
            d = _levenshtein(w, brand)
            denom = max(len(_norm(w)), len(_norm(brand))) or 1
            lev_conf = max(0.0, 1.0 - d/denom)           # 1.0 = identical
            sm_conf = _ratio(w, brand)                    # SequenceMatcher
            s = max(lev_conf, sm_conf)
            if s > best_s:
                best_s, best_b = s, brand
        if best_b and best_s >= threshold:
            length_wt = min(len(w_clean)/8.0, 1.0)        # longer tokens carry more
            near_perfect = 0.2 if best_s >= 0.90 else 0.0
            votes[best_b] += best_s * (0.6 + 0.4*length_wt) + near_perfect

    # Boost plant brand hint if present
    if plant_brand_hint:
        # gentle boost that can break ties / correct noise
        votes[plant_brand_hint] = votes.get(plant_brand_hint, 0.0) + 1.25

    # pick top brand by votes
    top_brand, top_score = max(votes.items(), key=lambda kv: kv[1])
    # tie-breaker: compare with full-text similarity if very close
    second = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[1] if len(votes) > 1 else (None, 0.0)
    if second[0] and (top_score - second[1] < 0.3):
        ft_top = _ratio(text, top_brand)
        ft_second = _ratio(text, second[0])
        if ft_second > ft_top + 0.05:
            top_brand = second[0]
            top_score = second[1]

    # normalise score into [0,1]-ish for reporting (heuristic)
    normalised = max(0.0, min(1.0, top_score / (top_score + 2.0)))
    if top_score <= 0.0 and plant_brand_hint:
        return plant_brand_hint, 0.6
    return (top_brand, normalised) if top_score > 0.0 else (None, 0.0)

# ------------------------------
# Model Detection
# ------------------------------
def _is_safety_line(line: str) -> bool:
    markers = (
        "WARNING","EXPLOSION","UNDERINFLATION","OVERLOADING","OWNER","MANUAL","VEHICLE",
        "IMPROPER","MOUNT","NEVER","EXCEED","PSI","KPA","LOAD","INJURY","RECOMMENDED","SEAT","BEADS"
    )
    up = line.upper()
    return sum(1 for m in markers if m in up) >= 2

def _brand_line_indices(text: str, manufacturer: str) -> List[int]:
    if not manufacturer:
        return []
    lines = text.splitlines()
    inds = []
    b = manufacturer
    for i, ln in enumerate(lines):
        if re.search(rf"(?i)\b{re.escape(b)}\b", ln) or re.search(rf"(?i)\b{re.escape(b.upper())}\b", ln):
            inds.append(i)
    # also respect fuzzy alias presence on a line
    for alias, canon in FUZZY_ALIASES.items():
        if canon == manufacturer:
            for i, ln in enumerate(lines):
                if re.search(rf"(?i)\b{re.escape(alias)}\b", ln):
                    inds.append(i)
    return sorted(set(inds))

def _window_lines(text: str, centers: List[int], before: int = 2, after: int = 3) -> str:
    lines = text.splitlines()
    keep: List[str] = []
    for c in centers:
        lo = max(0, c - before)
        hi = min(len(lines), c + after + 1)
        keep.extend(lines[lo:hi])
    return "\n".join(keep)

def _tokens_model_like(text: str) -> List[str]:
    """Tokens that *look* like model names: start with a letter, allow digits/dashes, len>=3, minus junk."""
    up = text.upper()
    toks = re.findall(r"\b[A-Z][A-Z0-9\-]{2,}\b", up)
    toks = [t for t in toks if t not in MODEL_STOPWORDS]
    toks = [t for t in toks if not SIZE_PAT.search(t)]
    toks = [t for t in toks if not LI_SPEED_PAT.search(t)]
    toks = [t for t in toks if not HOMO_PAT.search(t)]
    return toks

def _phrase_candidates(tokens: List[str], max_n: int = 3) -> List[str]:
    out: List[str] = []
    n = len(tokens)
    for k in range(n):
        for m in range(1, max_n + 1):
            if k + m > n: break
            phrase_tokens = tokens[k:k+m]
            if any(SIZE_PAT.search(t) or LI_SPEED_PAT.search(t) for t in phrase_tokens):
                continue
            phrase = " ".join(phrase_tokens).upper()
            # normalise common phrases
            if phrase in PHRASE_NORMALISERS:
                out.append(PHRASE_NORMALISERS[phrase])
            else:
                out.append(phrase)
    # de-dup preserving order
    seen = set(); uniq=[]
    for p in out:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

def _demote_if_tech(phrase: str) -> int:
    p = phrase.upper().replace(" ", "")
    return 1 if (p in {t.replace("-", "").replace(" ", "") for t in TECH_TAGS}) else 0

def _penalise_if_size_load(phrase: str) -> int:
    up = phrase.upper()
    return 1 if (SIZE_PAT.search(up) or LI_SPEED_PAT.search(up)) else 0

def _has_vowel(word: str) -> bool:
    return bool(re.search(r"[AEIOU]", word.upper()))

def _load_model_db() -> Dict[str, List[str]]:
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

def detect_model(text: str, manufacturer: str) -> str:
    """
    Model extraction:
      1) brand_models.json (or FALLBACK_MODELS) across the full OCR.
      2) tokens/phrases from lines near the brand (±2…+3 lines).
      3) brand-centered DOTALL window (±140 chars).
      4) whole-text fallback.
      Scoring tuple:
        (frequency, near-brand bonus, -safety_pen, -tech_pen, -size_load_pen, has_digit, vowel_bonus, length)
    """
    if not text:
        return ""

    brand = (manufacturer or "").strip()
    brand_up = brand.upper()

    # 1) Dictionary-first across the entire OCR text
    dict_list = MODELS.get(brand_up) or FALLBACK_MODELS.get(brand_up, [])
    if dict_list:
        hits = []
        for m in dict_list:
            if not m: continue
            pat = r"\b" + re.escape(m).replace(r"\ ", r"\s+") + r"\b"
            if re.search(pat, text, flags=re.IGNORECASE):
                hits.append(m)
            # also try normalised phrase variants (e.g., "MULTI ROUTE")
            for k, v in PHRASE_NORMALISERS.items():
                if v.upper() == m.upper():
                    if re.search(rf"\b{k}\b", text, flags=re.IGNORECASE):
                        hits.append(v)
        if hits:
            with_digit = [h for h in hits if re.search(r"\d", h)]
            choice = (max(with_digit, key=len) if with_digit else max(hits, key=len))
            m2 = re.search(rf"\b({re.escape(choice)})\b", text, flags=re.IGNORECASE)
            return m2.group(1).strip() if m2 else choice.title()

    # 2) Line-aware region around brand
    centers = _brand_line_indices(text, brand) if brand else []
    lines = text.splitlines()
    region = _window_lines(text, centers) if centers else ""
    region_tokens = _tokens_model_like(region) if region else []

    # 3) Brand-centered DOTALL window
    if not region_tokens and brand:
        m = re.search(rf"(?is)(.{{0,140}})\b{re.escape(brand)}\b(.{{0,140}})", text)
        if not m:
            m = re.search(rf"(?is)(.{{0,140}})\b{re.escape(brand_up)}\b(.{{0,140}})", text)
        if m:
            win = (m.group(1) or "") + " " + brand + " " + (m.group(2) or "")
            region_tokens = _tokens_model_like(win)

    # 4) Whole-text fallback
    full_tokens = _tokens_model_like(text)
    tokens = region_tokens or full_tokens

    # Build phrases (1–3 grams)
    phrases = _phrase_candidates(tokens, max_n=3)
    phrases = [p for p in phrases if _norm(p) != _norm(brand_up)]
    if not phrases:
        return ""

    # Pre-compute safety line indices
    safety_lines_idx = {i for i, ln in enumerate(lines) if _is_safety_line(ln)}

    # Score phrases
    full_up = text.upper()
    brand_near_up = region.upper() if region else ""
    scored: List[Tuple[Tuple, str]] = []

    for ph in sorted(set(phrases), key=len, reverse=False):
        ph_re = rf"\b{re.escape(ph)}\b"
        freq = len(re.findall(ph_re, full_up))
        near = 1 if (brand_near_up and re.search(ph_re, brand_near_up)) else 0

        safety_pen = 0
        for i, ln in enumerate(lines):
            if i in safety_lines_idx and re.search(ph_re, ln.upper()):
                safety_pen += 1

        tech_pen = _demote_if_tech(ph)
        size_pen = _penalise_if_size_load(ph)
        has_digit = 1 if re.search(r"\d", ph) else 0
        vowel_bonus = 1 if _has_vowel(ph) else 0
        score = (freq, near, -safety_pen, -tech_pen, -size_pen, has_digit, vowel_bonus, len(ph))
        scored.append((score, ph))

    scored.sort(reverse=True)
    best = scored[0][1]

    # If winner is a pure tech tag and we have other non-tech candidates, switch
    if _demote_if_tech(best):
        for sc, ph in scored:
            if not _demote_if_tech(ph):
                best = ph
                break

    # Normalise known phrases
    for k, v in PHRASE_NORMALISERS.items():
        if best.upper() == k:
            best = v
            break

    # Preserve OCR casing if present
    m2 = re.search(rf"\b({re.escape(best)})\b", text, flags=re.IGNORECASE)
    return m2.group(1).strip() if m2 else best.title()

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

# ------------------------------
# Optional CLI test
# ------------------------------
if __name__ == "__main__":
    import sys, json as _json
    s = sys.stdin.read() if not sys.argv[1:] else open(sys.argv[1], "r", encoding="utf-8").read()
    b, sc, m = extract_brand_and_model(s)
    print(_json.dumps({"brand": b, "score": round(sc,3), "model": m}, indent=2))