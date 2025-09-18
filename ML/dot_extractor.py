# dot_extractor.py
# Date-first, line-aware DOT/TIN extractor for tyre OCR.
# - Robust 4-digit WWYY detection (incl. sliding windows in tokens)
# - Strong preference for WWYY on DOT line and after DOT token
# - Small bracket bonus; penalize size/load tokens (e.g., 235/65R16C, 115/113R) and unit/pressure lines
# - Plant optional (2–3 alnum after DOT on DOT line); metadata lookup from JSON if available
# - No rejection of old tyres (YY 00..99). Tiny tie-break penalty for far-future YY.
# - Returns a compact result plus _debug scoring info.

from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ===============================
# Config / constants
# ===============================

DATE_WINDOW_SPAN_CHARS = 260          # extra char-window for "near DOT" (secondary)
PLANT_MAX_LEN = 3                     # plant: 2–3 alnum
PLANT_MIN_LEN = 2
SERIAL_MAX_LEN = 10                   # middle block max length
FUTURE_YEAR_SOFT_TOLERANCE = 5        # YY > now+5 gets small penalty

_UNIT_NOISE = {"LB", "LBS", "KG", "KGS", "KPA", "PSI", "BAR"}
_CONTEXT_NOISE = {"MAX", "LOAD", "INFLATION", "PRESSURE", "WEIGHT"}
_NOISE = _UNIT_NOISE | _CONTEXT_NOISE | {
    "RADIAL","TUBELESS","OUTSIDE","SAFETY","WARNING","MADE","IN","SIDEWALL","TREAD",
    "POLYESTER","NYLON","STEEL","PLY","PLIES","TEMPERATURE","TRACTION","TREADWEAR",
    "RIM","RIMS","STANDARD","MOUNT","ONLY","APPROVED","FOR","CAUTION","EXPLOSION",
    "OWNER","MANUAL","VEHICLE","FOLLOW","DOT"
}

# OCR digit repair map (for digit-mode cleaning)
_OCR_DIGIT_MAP = str.maketrans({
    "O":"0","o":"0",
    "I":"1","i":"1","L":"1","l":"1","(":"1",
    "Z":"2","z":"2",
    "S":"5","s":"5",
    "B":"8","b":"8",
})

# ===============================
# Plant metadata
# ===============================

_PLANT_MAP: Dict[str, Dict[str, str]] = {}

def _load_plant_codes() -> None:
    """
    Load plant metadata from:
      1) env: PLANT_CODES_JSON
      2) /mnt/data/plant_codes.json
      3) data/plant_codes.json

    Formats:
      dict: {"16C": {"manufacturer": "...", "plant": "...", "country": "..."}}
      list: [{"code":"16C","manufacturer":"...","plant":"...","country":"..."}]
    """
    global _PLANT_MAP
    paths: List[str] = []
    env_path = os.environ.get("PLANT_CODES_JSON")
    if env_path:
        paths.append(env_path)
    paths.extend(("/mnt/data/plant_codes.json", "data/plant_codes.json"))

    _PLANT_MAP.clear()
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    code = str(k).upper().strip()
                    _PLANT_MAP[code] = {
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
            if _PLANT_MAP:
                return
        except Exception:
            continue

def _lookup_plant(code: str) -> Dict[str, str]:
    code = (code or "").upper().strip()
    return _PLANT_MAP.get(code, {})

_load_plant_codes()

# ===============================
# Utilities
# ===============================

def _clean(tok: str, digits_mode: bool = False) -> str:
    """
    Clean token:
      - strip surrounding ()[]{}
      - remove / . -
      - uppercase
      - optional OCR digit repairs
      - keep only A-Z0-9
    """
    t = tok.strip()
    t = re.sub(r"^[\(\[\{]+|[\)\]\}]+$", "", t)
    t = re.sub(r"[\/\.\-]+", "", t)
    t = t.upper()
    if digits_mode:
        t = t.translate(_OCR_DIGIT_MAP)
    return re.sub(r"[^A-Z0-9]", "", t)

def _tokenize_window(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9\(\)\[\]\-/\.]+", text)

def _digit_windows(token: str) -> List[str]:
    """All 4-digit substrings from numeric runs in token."""
    ds = re.findall(r"\d+", token)
    out: List[str] = []
    for d in ds:
        if len(d) == 4:
            out.append(d)
        elif len(d) > 4:
            for i in range(0, len(d) - 3):
                out.append(d[i:i+4])
    return out

def _is_unit_context(tokens_clean: List[str], idx: int) -> bool:
    lo, hi = max(0, idx - 3), min(len(tokens_clean), idx + 4)
    window = tokens_clean[lo:hi]
    return any(t in _UNIT_NOISE or t in _CONTEXT_NOISE for t in window)

def _week_ok(w: int) -> bool:
    return 1 <= w <= 53

def _year_penalty(yy: int) -> float:
    """Soft tie-break penalty for far-future YY; old years are allowed."""
    now_yy = int(datetime.now().strftime("%y"))
    return 0.5 if yy > now_yy + FUTURE_YEAR_SOFT_TOLERANCE else 0.0

# ---------- line-aware tokenization ----------

@dataclass
class LineTokens:
    raw: List[List[str]]         # per-line raw tokens
    clean: List[List[str]]       # per-line clean-letter tokens
    digits: List[List[str]]      # per-line digit-repaired tokens
    flat_raw: List[str]          # flattened sequences
    flat_clean: List[str]
    flat_digits: List[str]
    map_global_to_line: List[Tuple[int, int]]   # global idx -> (line_idx, pos_in_line)
    dot_lines: Dict[int, int]    # line_idx -> first DOT pos
    unit_lines: set              # line indices that look like unit/pressure/load lines

def _tokenize_lines(text: str) -> LineTokens:
    lines = text.splitlines()
    raw_per = [ _tokenize_window(ln) for ln in lines ]
    clean_per = [[ _clean(t, False) for t in row if t.strip() ] for row in raw_per]
    digits_per = [[ _clean(t, True)  for t in row if t.strip() ] for row in raw_per]

    flat_raw: List[str] = []
    flat_clean: List[str] = []
    flat_digits: List[str] = []
    map_gl: List[Tuple[int, int]] = []

    for li, row in enumerate(raw_per):
        ci = [ _clean(t, False) for t in row if t.strip() ]
        di = [ _clean(t, True)  for t in row if t.strip() ]
        # keep alignment with original row order (we keep tokens even if clean is empty but numeric inside)
        k = 0
        for pi, tok in enumerate(row):
            # we still append; alignment bases on the filtered lists above
            ct = _clean(tok, False)
            dt = _clean(tok, True)
            # keep tokens that yield at least something clean or have digits
            if ct or re.search(r"\d", tok):
                flat_raw.append(tok)
                flat_clean.append(ct or tok)
                flat_digits.append(dt or tok)
                map_gl.append((li, pi))
                k += 1

    # Identify DOT lines and first DOT position
    dot_lines: Dict[int, int] = {}
    for li, row in enumerate(clean_per):
        for pi, ct in enumerate(row):
            if ct == "DOT":
                dot_lines.setdefault(li, pi)
                break

    # Identify likely unit/pressure/load lines (non-DOT diagnostic)
    unit_lines = set()
    for li, row in enumerate(clean_per):
        if li in dot_lines:
            continue
        tokens_set = set(row)
        if tokens_set & (_UNIT_NOISE | _CONTEXT_NOISE):
            unit_lines.add(li)

    return LineTokens(
        raw=raw_per, clean=clean_per, digits=digits_per,
        flat_raw=flat_raw, flat_clean=flat_clean, flat_digits=flat_digits,
        map_global_to_line=map_gl, dot_lines=dot_lines, unit_lines=unit_lines
    )

# ===============================
# Date candidate scoring (line-aware)
# ===============================

@dataclass
class DateCandidate:
    token_idx: int
    wwyy: str
    week: Optional[int]
    year: Optional[int]
    score: float
    reasons: List[str]

def _windows_after_dot_chars(text: str, span: int = DATE_WINDOW_SPAN_CHARS) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for m in re.finditer(r"(?i)\bDOT\b", text):
        start = m.end()
        spans.append((start, min(len(text), start + span)))
    return spans

def _score_date_candidates(text: str, L: LineTokens, dot_spans: List[Tuple[int, int]]) -> List[DateCandidate]:
    candidates: List[DateCandidate] = []

    # Build char offsets for "near DOT" (secondary proximity) over digits-joined
    joined = " ".join(L.flat_digits)
    offs: List[Tuple[int, int]] = []
    pos = 0
    for t in L.flat_digits:
        start = pos
        end = start + len(t)
        offs.append((start, end))
        pos = end + 1

    def near_dot(idx: int) -> bool:
        if not offs or not dot_spans:
            return False
        ts, te = offs[idx]
        return any((ts >= ds and ts <= de) or (te >= ds and te <= de) for (ds, de) in dot_spans)

    size_like = re.compile(r"\b\d{3}/\d{2}R\d{2}[A-Z0-9]?\b")  # 235/65R16C
    loadidx_like = re.compile(r"\b\d{2,3}/\d{2,3}\b")          # 115/113
    has_slash = re.compile(r"/")

    def penalize_size_load(raw_tok: str, clean_tok: str) -> bool:
        rt = raw_tok.upper()
        if size_like.search(rt): return True
        if loadidx_like.search(rt): return True
        if has_slash.search(rt): return True
        if rt.strip().endswith("R"): return True
        return False

    def is_bracketed(raw_tok: str, wwyy: str) -> bool:
        return bool(re.search(rf"[\(\[\{{]\s*{wwyy}\s*[\)\]\}}]", raw_tok))

    def later_bonus(idx: int, total: int) -> float:
        if total <= 1: return 0.0
        frac = idx / (total - 1)
        return 0.8 * frac  # light nudge only

    # Scan tokens for candidates
    for i, (tc, td) in enumerate(zip(L.flat_clean, L.flat_digits)):
        raw_tok = L.flat_raw[i]
        line_idx, pos_in_line = L.map_global_to_line[i]

        # Is this a DOT line, and where is DOT on it?
        on_dot_line = line_idx in L.dot_lines
        dot_pos = L.dot_lines.get(line_idx, -1)

        # 1) exact 4-digit token
        if re.fullmatch(r"\d{4}", td):
            wwyy = td
            wk, yy = int(wwyy[:2]), int(wwyy[2:])
            score = 0.0
            reasons: List[str] = []

            # Line-aware boosts
            if on_dot_line:
                score += 4.5; reasons.append("on DOT line")
                if dot_pos >= 0 and pos_in_line > dot_pos:
                    score += 2.5; reasons.append("after DOT on line")
            elif near_dot(i):
                score += 1.8; reasons.append("near DOT")

            # Small bracket bonus
            if is_bracketed(raw_tok, wwyy):
                score += 0.7; reasons.append("bracketed")

            # Token type bonus
            score += 0.8; reasons.append("token==WWYY")

            # Valid week
            if _week_ok(wk):
                score += 1.5; reasons.append("valid week")
            else:
                reasons.append("week out of range")

            # Later token light preference
            lb = later_bonus(i, len(L.flat_clean))
            score += lb; reasons.append(f"later {lb:.2f}")

            # Penalties
            if _is_unit_context(L.flat_clean, i):
                score -= 1.2; reasons.append("unit context")
            if (line_idx in L.unit_lines) and (not on_dot_line):
                score -= 2.0; reasons.append("unit line")
            if penalize_size_load(raw_tok, tc):
                score -= 2.5; reasons.append("size/load token")

            score -= _year_penalty(yy)

            candidates.append(DateCandidate(i, wwyy, wk if _week_ok(wk) else None, yy, score, reasons))

        # 2) sliding windows inside token
        for wwyy in _digit_windows(td):
            wk, yy = int(wwyy[:2]), int(wwyy[2:])
            score = 0.0
            reasons: List[str] = []

            if on_dot_line:
                score += 4.0; reasons.append("on DOT line")
                if dot_pos >= 0 and pos_in_line > dot_pos:
                    score += 2.2; reasons.append("after DOT on line")
            elif near_dot(i):
                score += 1.5; reasons.append("near DOT")

            if is_bracketed(raw_tok, wwyy):
                score += 0.5; reasons.append("bracketed")  # smaller than exact-token case

            score += 0.2; reasons.append("embedded WWYY")

            if _week_ok(wk):
                score += 1.5; reasons.append("valid week")
            else:
                reasons.append("week out of range")

            lb = later_bonus(i, len(L.flat_clean))
            score += lb; reasons.append(f"later {lb:.2f}")

            if _is_unit_context(L.flat_clean, i):
                score -= 1.2; reasons.append("unit context")
            if (line_idx in L.unit_lines) and (not on_dot_line):
                score -= 2.0; reasons.append("unit line")
            if penalize_size_load(raw_tok, tc):
                score -= 2.5; reasons.append("size/load token")

            score -= _year_penalty(yy)

            candidates.append(DateCandidate(i, wwyy, wk if _week_ok(wk) else None, yy, score, reasons))

    candidates.sort(key=lambda c: (c.score, c.token_idx), reverse=True)
    return candidates

# ===============================
# Plant extraction (optional)
# ===============================

def _find_plant_after_dot_lines(L: LineTokens) -> Tuple[str, int]:
    """
    Find a 2–3 char alnum plant right after DOT on the DOT line.
    Returns (plant_code, global_token_index_of_plant) or ("", -1).
    """
    for li, first_dot_pos in L.dot_lines.items():
        row_raw = L.raw[li]
        # find raw index of first DOT on the line
        raw_clean = [ _clean(t, False) for t in row_raw ]
        try:
            p_dot = raw_clean.index("DOT")
        except ValueError:
            p_dot = first_dot_pos

        look = range(p_dot + 1, min(p_dot + 6, len(row_raw)))
        # single-token
        for pi in look:
            cand_letters = _clean(row_raw[pi], False)
            cand_digits  = _clean(row_raw[pi], True)
            for cand in (cand_letters, cand_digits):
                if PLANT_MIN_LEN <= len(cand) <= PLANT_MAX_LEN and cand.isalnum():
                    # map line-local (li, pi) to global index
                    for gi, (g_li, g_pi) in enumerate(L.map_global_to_line):
                        if g_li == li and g_pi == pi:
                            return cand[:PLANT_MAX_LEN], gi
        # two-token combos (defensive)
        repaired = [(pi, _clean(row_raw[pi], True)) for pi in look]
        repaired = [(pi, s) for pi, s in repaired if s]
        if len(repaired) >= 2:
            combo = (repaired[0][1] + repaired[1][1])[:PLANT_MAX_LEN]
            if PLANT_MIN_LEN <= len(combo) <= PLANT_MAX_LEN and combo.isalnum():
                # map to global
                first_pi = repaired[0][0]
                for gi, (g_li, g_pi) in enumerate(L.map_global_to_line):
                    if g_li == li and g_pi == first_pi:
                        return combo, gi
    return "", -1

def _concat_middle(tokens_clean: List[str], start_idx: int, end_idx: int) -> str:
    mids: List[str] = []
    for k in range(start_idx, end_idx):
        t = tokens_clean[k]
        if not t or t in _NOISE or t == "DOT":
            continue
        mids.append(t)
    return "".join(mids)[:SERIAL_MAX_LEN]

# ===============================
# Public API
# ===============================

def parse_tin(text: str) -> Dict[str, object]:
    """
    Parse OCR text into DOT/TIN:
      {
        "full": "DOT <Plant?> <Serial?> <WWYY>",
        "parts": {
          "Plant": "...", "PlantName": "...", "PlantCountry": "...", "PlantManufacturer": "",
          "Serial": "<middle?>",
          "Date": "WWYY", "Week": "WW", "Year": "YY"
        },
        "_debug": { "date_candidates": [...], "chosen_from_dot_line": bool }
      }
    """
    out = {
        "full": "",
        "parts": {
            "Plant": "", "PlantName": "", "PlantCountry": "", "PlantManufacturer": "",
            "Serial": "", "Date": "", "Week": "", "Year": ""
        },
        "_debug": {
            "date_candidates": [],
            "chosen_from_dot_line": False
        }
    }
    if not text:
        return out

    # Tokenize per line and flatten (ensures alignment with line mapping)
    L = _tokenize_lines(text)
    if not L.flat_raw:
        return out

    dot_spans = _windows_after_dot_chars(text, DATE_WINDOW_SPAN_CHARS)

    # Score and choose best WWYY (date first!)
    candidates = _score_date_candidates(text, L, dot_spans)
    out["_debug"]["date_candidates"] = [
        {
            "token_idx": c.token_idx, "wwyy": c.wwyy,
            "week": c.week, "year": c.year,
            "score": round(c.score, 3), "reasons": c.reasons
        } for c in candidates[:12]
    ]
    if not candidates:
        return out

    best = candidates[0]
    wwyy = best.wwyy
    week = wwyy[:2]
    year = wwyy[2:]
    # chosen from dot line?
    li, _ = L.map_global_to_line[best.token_idx]
    out["_debug"]["chosen_from_dot_line"] = (li in L.dot_lines)

    # Optional plant
    plant, plant_gidx = _find_plant_after_dot_lines(L)

    # Serial/middle: tokens before date (global indices)
    before_clean = L.flat_clean[:best.token_idx]
    serial = ""
    if plant and plant in before_clean:
        p_idx = before_clean.index(plant)
        serial = _concat_middle(before_clean, p_idx + 1, len(before_clean))
    else:
        tail = before_clean[-2:] if len(before_clean) >= 2 else before_clean
        serial = _concat_middle(tail, 0, len(tail))

    # Human string
    blocks: List[str] = []
    if re.search(r"(?i)\bDOT\b", text): blocks.append("DOT")
    if plant: blocks.append(plant)
    if serial: blocks.append(serial)
    blocks.append(wwyy)
    full_str = " ".join(blocks).strip()

    # Plant meta
    plant_meta = _lookup_plant(plant) if plant else {}

    out["full"] = full_str
    out["parts"].update({
        "Plant": plant,
        "PlantName": plant_meta.get("plant", ""),
        "PlantCountry": plant_meta.get("country", ""),
        "PlantManufacturer": plant_meta.get("manufacturer", ""),
        "Serial": serial,
        "Date": wwyy,
        "Week": week,
        "Year": year
    })
    return out