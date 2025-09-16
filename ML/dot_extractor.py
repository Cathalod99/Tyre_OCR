# dot_extractor.py
# Offline TIN/DOT extractor: robust parsing + plant lookup.
from __future__ import annotations
import re
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ---------- Plant lookup ----------
_PLANT_MAP: Dict[str, Dict[str, str]] = {}

def _load_plant_codes() -> None:
    """
    Load plant metadata from either plant_codes.lookup_plant OR /mnt/data/plant_codes.json.
    Supports:
      - module: plant_codes.lookup_plant(code) -> dict or None
      - json (dict): {"1P2": {"manufacturer": "...", "plant": "...", "country": "..."}}
      - json (list): [{"code": "1P2", "manufacturer": "...", "plant": "...", "country": "..."}]
    """
    global _PLANT_MAP
    try:
        # Prefer local helper if available
        from plant_codes import lookup_plant as _lookup
        # Wrap to unify access pattern
        class _Wrapper:
            @staticmethod
            def get(code: str):
                try:
                    return _lookup(code) or None
                except Exception:
                    return None
        _PLANT_MAP["__MOD__"] = {"__module__": "1"}  # marker to use module
        _PLANT_MAP["__LOOKUP__"] = _Wrapper  # stash callable
        return
    except Exception:
        pass

    # Fallback to JSON
    json_path = "/mnt/data/plant_codes.json"
    if not os.path.exists(json_path):
        json_path = "data/plant_codes.json"  # secondary fallback

    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    k2 = str(k).upper().strip()
                    if not k2:
                        continue
                    _PLANT_MAP[k2] = {
                        "manufacturer": str(v.get("manufacturer", "")).strip(),
                        "plant": str(v.get("plant", "")).strip(),
                        "country": str(v.get("country", "")).strip(),
                    }
            elif isinstance(data, list):
                for row in data:
                    code = str(row.get("code", "")).upper().strip()
                    if not code:
                        continue
                    _PLANT_MAP[code] = {
                        "manufacturer": str(row.get("manufacturer", "")).strip(),
                        "plant": str(row.get("plant", "")).strip(),
                        "country": str(row.get("country", "")).strip(),
                    }
        except Exception:
            _PLANT_MAP = {}

_load_plant_codes()

def _lookup_plant(code: str) -> Dict[str, str]:
    code = (code or "").upper().strip()
    if not code:
        return {}
    if "__LOOKUP__" in _PLANT_MAP:
        # Use module adapter if present
        meta = _PLANT_MAP["__LOOKUP__"].get(code)
        if isinstance(meta, dict):
            return {
                "manufacturer": str(meta.get("manufacturer", "")).strip(),
                "plant": str(meta.get("plant", "")).strip(),
                "country": str(meta.get("country", "")).strip(),
            }
        return {}
    return _PLANT_MAP.get(code, {})

# ---------- Utils ----------
_UNIT_NOISE = {"LB", "LBS", "KG", "KGS", "KPA", "PSI"}
_CONTEXT_NOISE = {"MAX", "LOAD", "INFLATION", "PRESSURE", "WEIGHT"}
_NOISE = _UNIT_NOISE | _CONTEXT_NOISE | {
    "RADIAL","TUBELESS","OUTSIDE","SAFETY","WARNING","MADE","IN","SIDEWALL","TREAD",
    "POLYESTER","NYLON","STEEL","PLY","PLIES","TEMPERATURE","TRACTION","TREADWEAR",
    "RIM","RIMS","STANDARD","MOUNT","ONLY","APPROVED","FOR","CAUTION","EXPLOSION",
    "OWNER","MANUAL","VEHICLE","FOLLOW","DOT"  # we’ll handle DOT via windows
}

_OCR_DIGIT_MAP = str.maketrans({
    "O": "0", "o": "0",
    "I": "1", "i": "1", "L": "1", "l": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "B": "8", "b": "8",
})

def _clean(tok: str, digits_mode: bool) -> str:
    t = tok.strip()
    t = re.sub(r"^[\(\[\{]+|[\)\]\}]+$", "", t)
    t = re.sub(r"[\/\.\-]+", "", t)
    t = t.upper()
    if digits_mode:
        t = t.translate(_OCR_DIGIT_MAP)
    return re.sub(r"[^A-Z0-9]", "", t)

def _windows_after_dot(text: str) -> List[Tuple[bool, str]]:
    wins: List[Tuple[bool, str]] = []
    for m in re.finditer(r"(?i)\bDOT\b", text):
        wins.append((True, text[m.end():m.end()+240]))
    wins.append((False, text))  # always include full text as fallback
    return wins

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
    # common OCR flips: 8↔3, 6↔5, 9↔4, 0→8
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

def _is_unit_context(clean_tokens: List[str], idx: int) -> bool:
    lo, hi = max(0, idx-3), min(len(clean_tokens), idx+4)
    for j in range(lo, hi):
        if clean_tokens[j] in _UNIT_NOISE or clean_tokens[j] in _CONTEXT_NOISE:
            return True
    return False

def _pick_date(tokens_digits: List[str], tokens_clean: List[str]) -> Optional[Tuple[int,int,int,str]]:
    # pass 1: 4-digit
    for i in range(len(tokens_digits)-1, -1, -1):
        t = tokens_digits[i]
        if re.fullmatch(r"\d{4}", t):
            wk, yr = t[:2], t[2:]
            wi = _repair_week(wk)
            if wi is None:
                continue
            yi = int(yr)
            if not _valid_year(yi):
                continue
            if _is_unit_context(tokens_clean, i):
                continue
            return i, wi, yi, f"{wi:02d}{yi:02d}"
    # pass 2: 8-digit (split)
    for i in range(len(tokens_digits)-1, -1, -1):
        t = tokens_digits[i]
        if re.fullmatch(r"\d{8}", t):
            for s in [(t[:2], t[2:4]), (t[4:6], t[6:8])]:
                wk, yr = s
                wi = _repair_week(wk)
                if wi is None:
                    continue
                yi = int(yr)
                if not _valid_year(yi):
                    continue
                if _is_unit_context(tokens_clean, i):
                    continue
                return i, wi, yi, f"{wi:02d}{yi:02d}"
    # pass 3: 3-digit (OCR drop)
    for i in range(len(tokens_digits)-1, -1, -1):
        t = tokens_digits[i]
        if re.fullmatch(r"\d{3}", t):
            wk, yr = t[:2], t[2:]
            wi = _repair_week(wk)
            if wi is None:
                continue
            yi = int(yr)
            if not _valid_year(yi):
                continue
            if _is_unit_context(tokens_clean, i):
                continue
            return i, wi, yi, f"{wi:02d}{yi:02d}"
    return None

def _take_candidate_tokens(raw_tokens: List[str]) -> List[str]:
    out = []
    for t in raw_tokens:
        ct = _clean(t, digits_mode=False)
        if not ct or ct in _NOISE:
            continue
        # keep short blocks that look like TIN bits
        if 2 <= len(ct) <= 8:
            out.append(ct)
    return out

def _choose_plant(before_tokens: List[str]) -> str:
    """
    Plant code is the first 3 characters immediately after "DOT".
    Look for "DOT" in the tokens and take the next token's first 3 characters.
    """
    for i, t in enumerate(before_tokens):
        if t.upper() == "DOT" and i + 1 < len(before_tokens):
            # Next token after "DOT" should be the plant code
            next_token = before_tokens[i + 1]
            td = _clean(next_token, digits_mode=True)
            if len(td) >= 3 and td[0].isdigit():
                return td[:3]  # Take first 3 characters
    return ""

def parse_tin(text: str) -> Dict[str, Dict[str, str]]:
    out = {
        "full": "",
        "parts": {
            "Plant": "", "PlantName": "", "PlantCountry": "", "PlantManufacturer": "",
            "Serial": "", "Date": "", "Week": "", "Year": ""
        }
    }
    if not text:
        return out

    best = None

    for had_dot, window in _windows_after_dot(text):
        raw = re.findall(r"[A-Za-z0-9\(\)\[\]\-/\.]+", window)
        if not raw:
            continue
        toks_clean = [_clean(t, digits_mode=False) for t in raw if t.strip()]
        toks_clean = [t for t in toks_clean if t]

        toks_digits = [_clean(t, digits_mode=True) for t in raw if t.strip()]
        toks_digits = [t for t in toks_digits if t]

        # index into toks_digits == index into toks_clean because both derive from raw with same filtering
        date_hit = _pick_date(toks_digits, toks_clean)
        if not date_hit:
            continue
        di, week, year, last4 = date_hit

        before = toks_clean[:di]
        # Plant selection strictly 3 chars + starts with digit
        plant = _choose_plant(before)

        # Serial (new format 3+6+4): after plant, look for a 6-char block (alnum)
        serial = ""
        if plant:
            # find first occurrence of plant in before sequence (using digits mapping to be resilient)
            plant_found_at = -1
            for i, t in enumerate(before):
                if _clean(t, digits_mode=True) == plant:
                    plant_found_at = i
                    break
            if plant_found_at != -1:
                # next token(s) likely the serial; prefer 6 chars
                for j in range(plant_found_at + 1, len(before)):
                    cand = _clean(before[j], digits_mode=True)
                    if 5 <= len(cand) <= 8:  # tolerate minor OCR
                        serial = cand[:6] if len(cand) >= 6 else cand
                        break

        full_blocks = []
        if had_dot:
            full_blocks.append("DOT")
        if plant:
            full_blocks.append(plant)
        if serial:
            full_blocks.append(serial)
        full_blocks.append(last4)

        plant_meta = _lookup_plant(plant) if plant else {}
        cand = {
            "score": (10 if had_dot else 0) + (3 if plant else 0) + (2 if serial else 0),
            "full": " ".join(full_blocks).strip(),
            "parts": {
                "Plant": plant,
                "PlantName": plant_meta.get("plant", ""),
                "PlantCountry": plant_meta.get("country", ""),
                "PlantManufacturer": plant_meta.get("manufacturer", ""),
                "Serial": serial,
                "Date": last4,
                "Week": f"{week:02d}",
                "Year": f"{year:02d}",
            }
        }
        if (best is None) or (cand["score"] > best["score"]):
            best = cand

    if best:
        return {"full": best["full"], "parts": best["parts"]}
    return out