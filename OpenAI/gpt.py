# Tire OCR post-processing: robust local parsing + optional GPT-5 refinement.
# Enriches TIN with Plant metadata via plant_codes.lookup_plant or /mnt/data/plant_codes.json.

from __future__ import annotations
import os
import re
import json
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

# =========================================================
# OpenAI client (supports both new >=1.x and legacy <1.0)
# =========================================================
_USE_OAI = False

def _mk_chat_func():
    """
    Returns a function chat(model, messages, temperature) -> str | None that
    yields the assistant message content, or None on failure.
    Works with openai>=1.0 (OpenAI client) and legacy openai<1.0.
    """
    # Try new SDK first
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI()  # reads OPENAI_API_KEY from env
        def _chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.1) -> Optional[str]:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=messages,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception:
                return None
        return True, _chat
    except Exception:
        pass

    # Try legacy SDK
    try:
        import openai  # type: ignore
        if not getattr(openai, "api_key", None):
            openai.api_key = os.getenv("OPENAI_API_KEY", "")
        def _chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.1) -> Optional[str]:
            try:
                resp = openai.ChatCompletion.create(
                    model=model,
                    temperature=temperature,
                    messages=messages,
                )
                return (resp["choices"][0]["message"]["content"] or "").strip()
            except Exception:
                return None
        return True, _chat
    except Exception:
        pass

    return False, lambda *args, **kwargs: None

_USE_OAI, _chat = _mk_chat_func()

# =========================================================
# Plant lookup (module helper or /mnt/data/plant_codes.json)
# =========================================================
def _load_plant_map() -> Dict[str, Dict[str, str]]:
    # Support both dict and array formats.
    # Preferred location: /mnt/data/plant_codes.json; fallback: data/plant_codes.json
    candidates = ["/mnt/data/plant_codes.json", "data/plant_codes.json", "./plant_codes.json"]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                plant_map: Dict[str, Dict[str, str]] = {}
                if isinstance(data, dict):
                    # { "1P2": {"manufacturer":...,"plant":...,"country":...}, ... }
                    for code, info in data.items():
                        code = str(code).upper().strip()
                        if not code:
                            continue
                        plant_map[code] = {
                            "manufacturer": str(info.get("manufacturer", "")).strip(),
                            "plant": str(info.get("plant", "")).strip(),
                            "country": str(info.get("country", "")).strip(),
                        }
                elif isinstance(data, list):
                    # [ {"code":"1P2","manufacturer":...,"plant":...,"country":...}, ... ]
                    for rec in data:
                        code = str(rec.get("code", "")).upper().strip()
                        if not code:
                            continue
                        plant_map[code] = {
                            "manufacturer": str(rec.get("manufacturer", "")).strip(),
                            "plant": str(rec.get("plant", "")).strip(),
                            "country": str(rec.get("country", "")).strip(),
                        }
                return plant_map
            except Exception:
                return {}
    return {}

_PLANT_MAP: Dict[str, Dict[str, str]] = _load_plant_map()

def _lookup_plant_local(code: str) -> Optional[Dict[str, str]]:
    code = (code or "").upper().strip()
    return _PLANT_MAP.get(code)

# Prefer module function if available
try:
    from plant_codes import lookup_plant as _lookup_plant_mod  # type: ignore
    def lookup_plant(code: str):
        return _lookup_plant_mod(code) or _lookup_plant_local(code)
except Exception:
    def lookup_plant(code: str):
        return _lookup_plant_local(code)

# =========================================================
# Static data
# =========================================================
BRAND_LIST: List[str] = [
    "APlus","AUSTONE","Alliance","Altenzo","Antares","Apollo","Arivo","Atlas","Autogreen","Avon",
    "BF Goodrich","Barum","Bridgestone","CST","Ceat","Challenger","Chengshan","Continental","Cooper",
    "Davanti","Dayton","Debica","Delinte","Diamondback","Diplomat","Double Star","Dunlop","Duraturn",
    "Dynamo","Event","Evergreen","Falken","Firemax","Firestone","Fortuna","Fortune","Fronway","Fulda",
    "GT Radial","Gislaved","Goodride","Goodyear","Greentrac","Grenlander","Habilead","Haida","Hankook",
    "llink","Infinity","lnsa Turbo","lnvovic","Iris","Kelly","Kenda","Kingboss","Kingstar","Kleber",
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

KNOWN_MODELS: Dict[str, List[str]] = {
    "MICHELIN": [r"\bPRIMACY\s*\d?\b", r"\bPILOT\s*SPORT\s*\d?\b", r"\bCROSSCLIMATE\s*\d?\b",
                 r"\bENERGY\s*SAVER\b", r"\bLATITUDE\b", r"\bALPIN\b"],
    "BRIDGESTONE": [r"\bTURANZA\b", r"\bPOTENZA\b", r"\bDUELER\b", r"\bBLIZZAK\b", r"\bECOPIA\b"],
    "CONTINENTAL": [r"\bPREMIUMCONTACT\s*\d?\b", r"\bSPORTCONTACT\s*\d?\b", r"\bECOCONTACT\s*\d?\b",
                    r"\bWINTERCONTACT\s*\d?\b", r"\bALLSEASONCONTACT\b"],
    "GOODYEAR": [r"\bEAGLE\s*F1\b", r"\bEFFICIENTGRIP\b", r"\bVECTOR\s*4SEASONS\b", r"\bULTRAGRIP\b", r"\bWRANGLER\b"],
    "PIRELLI": [r"\bP\s*ZERO\b", r"\bCINTURATO\b", r"\bSCORPION\b", r"\bSOTTOZERO\b", r"\bCARRIER\b"],
    "HANKOOK": [r"\bVENTUS\b", r"\bKINERGY\b", r"\bDYNAPRO\b", r"\bWINTER\s*I\*?CEPT\b"],
    "YOKOHAMA": [r"\bADVAN\b", r"\bBLUEARTH\b", r"\bGEOLANDAR\b", r"\bICEGUARD\b"],
    "DUNLOP": [r"\bSP\s*SPORT\b", r"\bWINTER\s*SPORT\b", r"\bGRANDTREK\b"],
    "TOYO": [r"\bPROXES\b", r"\bOPEN\s*COUNTRY\b", r"\bOBSERVE\b", r"\bNANOENERGY\b"],
    "NOKIAN": [r"\bHAKKAPELIITTA\b", r"\bWR\s*SNOWPROOF\b", r"\bNORDMAN\b"],
}

MODEL_STOPWORDS = {
    "TOTAL","PERFORMANCE","TREAD","TREADWEAR","TRACTION","TEMPERATURE","RADIAL","TUBELESS",
    "OUTSIDE","EXTRA","LOAD","WARNING","MAX","PRESS","MADE","IN","POLYESTER","POLYAMIDE",
    "STEEL","SIDEWALL","PLY","ROTATION","TEMPERATORE","SAFETY","M+S","XL"
}

# =========================================================
# Helpers
# =========================================================
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

def best_brand_match(text: str, threshold: float = 0.30) -> Tuple[Optional[str], float]:
    for brand in BRAND_LIST:
        pat = r"\b" + re.escape(brand).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pat, text, re.IGNORECASE):
            return brand, 1.0
    scores = [(brand, _ratio(text, brand)) for brand in BRAND_LIST]
    best = max(scores, key=lambda x: x[1])
    return (best[0], best[1]) if best[1] >= threshold else (None, 0.0)

def parse_tire_size(text: str) -> Optional[str]:
    # 205/55R16, 215/50 ZR17, 255/45 R20 etc.
    m = re.search(r"\b(\d{3})\s*/\s*(\d{2})\s*[Zz]?[Rr]\s*(\d{2})\b", text)
    return f"{m.group(1)}/{m.group(2)}R{m.group(3)}" if m else None

def parse_li_speed(text: str) -> Optional[str]:
    for m in re.finditer(r"\b(\d{2,3})\s*([A-Z])\b", text):
        li_s, sp = m.group(1), m.group(2).upper()
        try:
            li = int(li_s)
        except ValueError:
            continue
        if not (60 <= li <= 130):
            continue
        ctx = text[max(0, m.start()-10):m.end()+10].upper()
        if any(u in ctx for u in ["PSI","KPA","LBS","LB","KG","KGS"]):
            continue
        if sp == "R" and re.match(r"\s*(1[0-9]|2[0-6])\b", text[m.end():m.end()+4]):
            continue
        return f"{li}/{sp}"
    return None

# =========================================================
# TIN parsing
# =========================================================
_UNIT_NOISE = {"LB","LBS","KG","KGS","KPA","PSI"}
_CONTEXT_NOISE = {"MAX","LOAD","INFLATION","PRESSURE","WEIGHT"}

def _windows_after_dot(text: str) -> List[Tuple[bool, str]]:
    """
    Create search windows: if 'DOT' appears, use slices after each 'DOT'.
    Always include a full-text window so DOT is optional.
    """
    wins: List[Tuple[bool, str]] = []
    for m in re.finditer(r"(?i)\bDOT\b", text):
        wins.append((True, text[m.end():m.end()+250]))
    wins.append((False, text))
    return wins

def _clean(tok: str, digits_mode: bool) -> str:
    t = tok.strip()
    t = re.sub(r"^[\(\[\{]+|[\)\]\}]+$", "", t)
    t = re.sub(r"[\/\.\-]+", "", t)
    t = t.upper()
    if digits_mode:
        # aggressive OCR-to-digit normalization for date/serial reading
        t = (t.replace("O","0").replace("I","1").replace("L","1")
               .replace("Z","2").replace("S","5").replace("B","8"))
    return re.sub(r"[^A-Z0-9]", "", t)

def _valid_week(w: int) -> bool:
    return 1 <= w <= 53

def _valid_year(y: int) -> bool:
    now_yy = int(datetime.now().strftime("%y"))
    return 0 <= y <= now_yy

def _repair_week(wk: str) -> Optional[int]:
    if len(wk) != 2 or not wk.isdigit():
        return None
    w = int(wk)
    if _valid_week(w):
        return w
    # Common OCR flips
    fixes = [("8","3"),("6","5"),("9","4"),("0","8")]
    cs = list(wk)
    for i in range(2):
        orig = cs[i]
        for a,b in fixes:
            if orig == a:
                cs[i] = b
                cand = int("".join(cs))
                if _valid_week(cand):
                    return cand
        cs[i] = orig
    return None

def _is_unit_context(raw_tokens: List[str], idx: int) -> bool:
    """True if tokens near idx include unit/weight/pressure words."""
    rng = range(max(0, idx-3), min(len(raw_tokens), idx+4))
    for j in rng:
        ct = _clean(raw_tokens[j], digits_mode=False)
        if ct in _UNIT_NOISE or ct in _CONTEXT_NOISE:
            return True
    return False

def _find_date(tokens: List[str], raw_tokens: List[str]) -> Optional[Tuple[int,int,int,str]]:
    """
    Find most likely 4-digit YY-coded date; ignore weight/pressure contexts.
    Returns (index, week, year, 'WWYY').
    """
    # Pass 1: 4 digits
    for i in range(len(tokens) - 1, -1, -1):
        t = tokens[i]
        if re.fullmatch(r"\d{4}", t):
            wk, yr = t[:2], t[2:]
            wk_i = _repair_week(wk)
            if wk_i is None:
                continue
            yr_i = int(yr)
            if not _valid_year(yr_i):
                continue
            if _is_unit_context(raw_tokens, i):
                continue
            return i, wk_i, yr_i, f"{wk_i:02d}{yr_i:02d}"

    # Pass 1.5: try splitting an 8-digit blob into two 4-digit candidates
    for i in range(len(tokens) - 1, -1, -1):
        t = tokens[i]
        if re.fullmatch(r"\d{8}", t):
            pairs = [(t[:2], t[2:4]), (t[4:6], t[6:8])]
            for wk, yr in pairs:
                try:
                    wk_i = _repair_week(wk)
                    if wk_i is None:
                        continue
                    yr_i = int(yr)
                    if not _valid_year(yr_i):
                        continue
                    if _is_unit_context(raw_tokens, i):
                        continue
                    return i, wk_i, yr_i, f"{wk_i:02d}{yr_i:02d}"
                except Exception:
                    continue

    # Pass 2: 3 digits (OCR dropped one)
    for i in range(len(tokens) - 1, -1, -1):
        t = tokens[i]
        if re.fullmatch(r"\d{3}", t):
            wk, yr = t[:2], t[2:]
            wk_i = _repair_week(wk)
            if wk_i is None:
                continue
            yr_i = int(yr)
            if not _valid_year(yr_i):
                continue
            if _is_unit_context(raw_tokens, i):
                continue
            return i, wk_i, yr_i, f"{wk_i:02d}{yr_i:02d}"
    return None

# ---- Plant code rules (new format): exactly 3 chars, must start with a digit
def _normalize_plant_strict(c: str) -> str:
    """Normalize plant code to exactly 3 chars; coerce leading OCR letter to digit if needed."""
    c = re.sub(r"[^A-Za-z0-9]", "", (c or "").upper())
    if len(c) < 3:
        return ""
    c = c[:3]
    # fix leading OCR misreads: T->1, O->0, I/L->1, S->5, Z->2, B->8
    lead = c[0]
    lead_map = {"T":"1","O":"0","I":"1","L":"1","S":"5","Z":"2","B":"8"}
    if not lead.isdigit():
        lead = lead_map.get(lead, lead)
    c = lead + c[1:]
    # final check
    return c if len(c) == 3 and c[0].isdigit() else ""

def _segment_codes(before_date_raw_tokens: List[str]) -> Tuple[str,str,str]:
    """
    Build plausible blocks before date.
    Heuristic:
      - Collect compact tokens; skip obvious noise.
      - The concatenated stream's FIRST 3 chars (after cleaning) → Plant (strict: 3 chars, starts with digit).
      - Next up to 6 chars → Size code (legacy), remainder up to 6 → Manufacturer code.
      - If we only have Plant + one 6-char block, treat it as serial/manufacturer (new 3+6+4 style).
    """
    NOISE = {"DOT","MAX","LOAD","INFLATION","PRESSURE","KPA","PSI","KG","KGS","LBS","LB",
             "RADIAL","TUBELESS","OUTSIDE","SAFETY","TEMPERATURE","TREAD","EXTRA"}
    # Clean and keep short-ish blocks
    kept: List[str] = []
    for t in before_date_raw_tokens:
        ct = _clean(t, digits_mode=False)
        if not ct or ct in NOISE:
            continue
        if 1 <= len(ct) <= 8:
            kept.append(ct)

    if not kept:
        return "", "", ""

    # Concatenate with no separators (to resist random splits)
    stream = "".join(kept)
    if not stream:
        return "", "", ""

    # Extract plant strictly from the FIRST 3 chars of stream
    plant_raw3 = stream[:3]
    plant = _normalize_plant_strict(plant_raw3)

    # Advance offset if plant was valid; otherwise, try to rescan by skipping first char
    offset = 3 if plant else 0
    size_code = ""
    manuf = ""

    # Remaining characters after plant
    rem = stream[offset:]

    # Legacy like 2-3 + 2-6 + 1-6; New like 3 + 6 + 4
    if rem:
        # Prefer a 6-char block as the next chunk (size_code for legacy OR serial)
        if len(rem) >= 6:
            nxt = rem[:6]
            # If we found only plant+6, treat this as manufacturer/serial; else try to split into size+manuf
            size_code = rem[:3] if len(rem) >= 9 else ""
            if size_code:
                manuf = rem[3:9][:6]
            else:
                manuf = nxt
        else:
            # Short tail: treat as size code only (legacy oddities)
            size_code = rem[:6]

    # Hard cap manufacturer to 6
    manuf = manuf[:6]

    # If plant invalid but we have at least 3 chars ahead, try sliding window of first 5 to find a valid 3-char
    if not plant and len(stream) >= 5:
        for i in range(0, 3):  # 0,1,2
            cand = _normalize_plant_strict(stream[i:i+3])
            if cand:
                plant = cand
                # Rebuild rem after this position
                rem = stream[i+3:]
                size_code = rem[:3] if len(rem) >= 9 else ""
                manuf = (rem[3:9] if size_code else rem[:6])[:6]
                break

    return plant, size_code, manuf

def parse_tin_full(text: str) -> Dict[str, Dict[str, str]]:
    out = {"full": "", "parts": {
        "Plant":"", "PlantName":"", "PlantCountry":"", "PlantManufacturer":"",
        "SizeCode":"", "ManufacturerCode":"", "Date":"", "Week":"", "Year":""
    }}
    if not text:
        return out

    best: Optional[Dict[str, Any]] = None

    for had_dot, win in _windows_after_dot(text):
        raw_tokens = re.findall(r"[A-Za-z0-9\(\)\[\]\-/\.]+", win)
        if not raw_tokens:
            continue

        digits_stream = [_clean(t, digits_mode=True) for t in raw_tokens if t.strip()]
        digits_stream = [t for t in digits_stream if t]

        date_hit = _find_date(digits_stream, raw_tokens)
        if not date_hit:
            continue

        di, week, year, last4 = date_hit
        plant, size_code, manuf = _segment_codes(raw_tokens[:di])

        # Enforce strict plant code rules
        plant = _normalize_plant_strict(plant)

        # Lookup plant meta
        plant_name = plant_country = plant_mfr = ""
        if plant:
            meta = lookup_plant(plant)
            if meta:
                plant_mfr = str(meta.get("manufacturer","")).strip()
                plant_name = str(meta.get("plant","")).strip()
                plant_country = str(meta.get("country","")).strip()

        blocks = [b for b in [plant, size_code, manuf] if b]
        full = ("DOT " if had_dot else "") + " ".join(blocks + [last4])

        cand = {
            "score": (10 if had_dot else 0) + (3 if plant else 0) + (1 if size_code else 0) + (1 if manuf else 0),
            "full": full.strip(),
            "parts": {
                "Plant": plant,
                "PlantName": plant_name,
                "PlantCountry": plant_country,
                "PlantManufacturer": plant_mfr,
                "SizeCode": size_code,
                "ManufacturerCode": manuf,
                "Date": last4,
                "Week": f"{week:02d}",
                "Year": f"{year:02d}",
            },
        }
        if (best is None) or (cand["score"] > best["score"]):
            best = cand

    if best:
        return {"full": best["full"], "parts": best["parts"]}
    return out

# =========================================================
# Model guesser
# =========================================================
def _is_known_model(brand: str, model: str) -> bool:
    if not brand or not model:
        return False
    pats = KNOWN_MODELS.get(brand.upper(), [])
    u = model.upper()
    return any(re.search(p, u) for p in pats)

def _guess_model(text: str, manufacturer: str) -> str:
    if not manufacturer:
        return ""
    u = text.upper()
    brand = manufacturer.upper()
    pats = KNOWN_MODELS.get(brand, [])
    hits = []
    for pat in pats:
        m = re.search(pat, u)
        if m:
            hits.append(m.group(0).title())
    if hits:
        with_digit = [h for h in hits if re.search(r"\d", h)]
        return (with_digit[0] if with_digit else max(hits, key=len)).strip()

    win = re.search(rf"(?i)\b{re.escape(manufacturer)}\b(.{{0,50}})", text)
    window = win.group(1) if win else text
    tokens = re.findall(r"\b[A-Z][A-Za-z0-9\-]+\b", window)
    tokens = [t for t in tokens if t.upper() not in MODEL_STOPWORDS and _norm(t) != _norm(manufacturer)]
    cands: List[Tuple[Tuple[int,int], str]] = []
    for n in (3, 2, 1):
        for i in range(len(tokens)-n+1):
            phrase = " ".join(tokens[i:i+n])
            score = (1 if re.search(r"\d", phrase) else 0, len(phrase))
            cands.append((score, phrase))
        if cands:
            break
    return (sorted(cands, reverse=True)[0][1] if cands else "").strip()

# =========================================================
# GPT-5 (optional refinement)
# =========================================================
SYSTEM_PROMPT = (
    "You are a Tire Data Extraction Specialist. Return ONLY valid JSON with keys: "
    '{"Manufacturer":"","Tire model":"","Tire size":"","Load index and speed rating":"",'
    '"Scan TIN":{"Full DOT":"","DOT Code":"","Week Code":"","Year Code":"",'
    '"Plant Code":"","Plant Name":"","Plant Country":"","Plant Manufacturer":""}}.'
)

def _extract_json(s: str) -> str:
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1 or b <= a:
        raise ValueError("No JSON object found.")
    return s[a:b+1]

def _call_gpt5(ocr_text: str, seed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not _USE_OAI:
        return None
    user_msg = (
        "Extract manufacturer, tire model, tire size (e.g., 205/55R16), "
        "and load index/speed (e.g., 102V → 102/V). For TIN:\n"
        " - Use 'DOT ' prefix ONLY if it appears in OCR near the code.\n"
        " - The final 4 digits are the DOT date code (WWYY).\n"
        " - IMPORTANT: New DOT plant code is EXACTLY 3 characters and MUST start with a digit.\n"
        "   Normalize OCR mistakes: leading T→1, O→0, I/L→1, S→5, Z→2, B→8.\n"
        " - Ignore 4-digit numbers that appear near weight/pressure contexts (LB/LBS/KG/KPA/PSI/MAX/LOAD/etc.).\n"
        "Return STRICT JSON only.\n\n"
        f"--- OCR TEXT ---\n{ocr_text}\n\n--- HINTS ---\n{json.dumps(seed, ensure_ascii=False)}"
    )
    try:
        resp = _chat(
            model="gpt-5",
            messages=[{"role":"system","content":SYSTEM_PROMPT},
                      {"role":"user","content":user_msg}],
            temperature=0.1,
        )
        if not resp:
            return None
        return json.loads(_extract_json(resp))
    except Exception:
        return None

# =========================================================
# Public API
# =========================================================
def get_tyre_info(ocr_text: str) -> Dict[str, Any]:
    size = parse_tire_size(ocr_text) or ""
    li_speed = parse_li_speed(ocr_text) or ""
    brand_guess, score = best_brand_match(ocr_text)
    manufacturer = (brand_guess or "").strip()
    local_model = _guess_model(ocr_text, manufacturer)

    # Local TIN parse
    tin = parse_tin_full(ocr_text)
    full_dot = tin["full"].strip()
    date = tin["parts"]["Date"]
    wk = tin["parts"]["Week"] or ""
    yr = tin["parts"]["Year"] or ""
    plant_code = tin["parts"]["Plant"] or ""
    plant_name = tin["parts"]["PlantName"] or ""
    plant_country = tin["parts"]["PlantCountry"] or ""
    plant_mfr = tin["parts"]["PlantManufacturer"] or ""

    seed = {
        "brand_guess": manufacturer, "brand_score": round((score or 0.0), 3),
        "tire_size_guess": size, "li_speed_guess": li_speed,
        "rules": {
            "plant_code_len": 3,
            "plant_code_start_digit": True,
            "ignore_weight_pressure_context": True
        },
        "tin_guess": {
            "Full DOT": full_dot, "DOT Code": date,
            "Week Code": wk, "Year Code": yr,
            "Plant Code": plant_code, "Plant Name": plant_name,
            "Plant Country": plant_country, "Plant Manufacturer": plant_mfr,
        },
        "brand_list": BRAND_LIST[:],
    }

    gpt = _call_gpt5(ocr_text, seed) or {}

    # Model choice
    gpt_model = (gpt.get("Tire model") or "").strip()
    if _is_known_model(manufacturer, local_model):
        final_model = local_model
    elif gpt_model and not any(w in gpt_model.upper().split() for w in MODEL_STOPWORDS):
        final_model = gpt_model
    else:
        final_model = local_model

    # Merge with validation
    def _pick(a: str, b: str) -> str:
        return (a or b or "").strip()

    merged: Dict[str, Any] = {
        "Manufacturer": _pick(gpt.get("Manufacturer",""), manufacturer),
        "Tire model": final_model,
        "Tire size": _pick(gpt.get("Tire size",""), size).replace(" ", "").upper(),
        "Load index and speed rating": _pick(gpt.get("Load index and speed rating",""), li_speed).upper(),
        "Scan TIN": {
            "Full DOT": _pick(gpt.get("Scan TIN", {}).get("Full DOT",""), full_dot),
            "DOT Code": _pick(gpt.get("Scan TIN", {}).get("DOT Code",""), date),
            "Week Code": _pick(gpt.get("Scan TIN", {}).get("Week Code",""), wk),
            "Year Code": _pick(gpt.get("Scan TIN", {}).get("Year Code",""), yr),
            "Plant Code": _pick(gpt.get("Scan TIN", {}).get("Plant Code",""), plant_code),
            "Plant Name": _pick(gpt.get("Scan TIN", {}).get("Plant Name",""), plant_name),
            "Plant Country": _pick(gpt.get("Scan TIN", {}).get("Plant Country",""), plant_country),
            "Plant Manufacturer": _pick(gpt.get("Scan TIN", {}).get("Plant Manufacturer",""), plant_mfr),
        },
    }

    # Normalize size
    m = re.match(r"^\s*(\d{3})\s*/\s*(\d{2})\s*[Zz]?[Rr]\s*(\d{2})\s*$", merged["Tire size"])  # noqa: W605
    if m:
        merged["Tire size"] = f"{m.group(1)}/{m.group(2)}R{m.group(3)}"

    # Normalize LI/Speed
    ms = re.match(r"^\s*(\d{2,3})\s*[/]?\s*([A-Z])\s*$", merged["Load index and speed rating"])  # noqa: W605
    if ms:
        merged["Load index and speed rating"] = f"{ms.group(1)}/{ms.group(2)}"

    # Validate & clamp DOT parts
    now_yy = int(datetime.now().strftime("%y"))
    wk_str = (merged["Scan TIN"]["Week Code"] or "").strip()
    yr_str = (merged["Scan TIN"]["Year Code"] or "").strip()
    dc = (merged["Scan TIN"]["DOT Code"] or "").strip()

    # If DOT Code present, it rules the week/year if valid
    if re.fullmatch(r"\d{4}", dc):
        wk2, yr2 = int(dc[:2]), int(dc[2:])
        if _valid_week(wk2):
            wk_str = f"{wk2:02d}"
        if _valid_year(yr2):
            yr_str = f"{yr2:02d}"

    try:
        wki = int(wk_str) if wk_str else 0
    except Exception:
        wki = 0
    try:
        yri = int(yr_str) if yr_str else -1
    except Exception:
        yri = -1

    merged["Scan TIN"]["Week Code"] = f"{wki:02d}" if (1 <= wki <= 53) else ""
    merged["Scan TIN"]["Year Code"] = f"{yri:02d}" if (0 <= yri <= now_yy) else ""
    merged["Scan TIN"]["DOT Code"] = (merged["Scan TIN"]["Week Code"] + merged["Scan TIN"]["Year Code"]) if (merged["Scan TIN"]["Week Code"] and merged["Scan TIN"]["Year Code"]) else ""

    # Enforce plant code rules (3 chars, start digit), with OCR normalization
    merged["Scan TIN"]["Plant Code"] = _normalize_plant_strict(merged["Scan TIN"]["Plant Code"])
    # If empty, try to salvage from local parse result
    if not merged["Scan TIN"]["Plant Code"]:
        merged["Scan TIN"]["Plant Code"] = _normalize_plant_strict(plant_code)

    # Refresh plant metadata if we’ve got a (possibly corrected) code
    if merged["Scan TIN"]["Plant Code"]:
        meta = lookup_plant(merged["Scan TIN"]["Plant Code"])
        if meta:
            merged["Scan TIN"]["Plant Manufacturer"] = str(meta.get("manufacturer","")).strip()
            merged["Scan TIN"]["Plant Name"] = str(meta.get("plant","")).strip()
            merged["Scan TIN"]["Plant Country"] = str(meta.get("country","")).strip()

    # If Full DOT seems to have a wrong plant block, rebuild it cleanly
    # Compose blocks we trust: Plant (strict) + (optional) SizeCode + ManufacturerCode + Date
    strict_blocks = []
    if merged["Scan TIN"]["Plant Code"]:
        strict_blocks.append(merged["Scan TIN"]["Plant Code"])
    if tin["parts"]["SizeCode"]:
        strict_blocks.append(tin["parts"]["SizeCode"])
    if tin["parts"]["ManufacturerCode"]:
        strict_blocks.append(tin["parts"]["ManufacturerCode"])
    if merged["Scan TIN"]["DOT Code"]:
        strict_blocks.append(merged["Scan TIN"]["DOT Code"])

    # Preserve DOT prefix only if present in OCR context window
    had_dot_any = any(re.search(r"(?i)\bDOT\b", w[1]) for w in _windows_after_dot(ocr_text))
    merged["Scan TIN"]["Full DOT"] = (("DOT " if had_dot_any else "") + " ".join(strict_blocks)).strip()

    # Snap manufacturer spelling to a known brand if close
    if merged["Manufacturer"]:
        snapped, _ = best_brand_match(merged["Manufacturer"])
        if snapped:
            merged["Manufacturer"] = snapped

    return merged

# =========================================================
# CLI test
# =========================================================
if __name__ == "__main__":
    sample = """
    DELINTE 245/40ZR18 97W M+S
    DOT 1P2 XEKUGC 0128
    MAX LOAD 1809 LBS
    """
    print(json.dumps(get_tyre_info(sample), indent=2))