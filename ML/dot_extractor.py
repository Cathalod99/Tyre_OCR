# dot_extractor.py
# Date-first, line-aware DOT/TIN extractor for tyre OCR.
# - Robust WWYY detection (4-digit, incl. sliding windows inside long tokens)
# - Strong preference for WWYY on DOT line and AFTER "DOT"; end-of-line nudge
# - Bracket bonus; heavy penalties for size/load/pressure lines
# - ACCEPTANCE GATE so we do NOT output a date unless it looks genuine:
#     * If "DOT" exists: require proximity to DOT (same line or near chars) AND min score
#     * If no "DOT": accept only clearly bracketed dates with higher score
# - Plant (optional): prefer 2–3 chars starting with 0/1/2 (new scheme);
#   accept OCR-fixed variants (O→0, I/L/(→1, etc.)
# - Old tyres allowed (YY 00..99). Small tie-break penalty for far-future YY.
# - If no acceptable date: STILL return plant/serial so truck tyres without WWYY are useful.
# - Returns compact result plus _debug info.

from __future__ import annotations
import os, re, json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ===============================
# Config / constants
# ===============================

DATE_WINDOW_SPAN_CHARS = 260          # secondary char-window proximity to DOT
PLANT_MAX_LEN = 3
PLANT_MIN_LEN = 2
SERIAL_MAX_LEN = 10
FUTURE_YEAR_SOFT_TOLERANCE = 5        # YY > now+5 gets small penalty

_UNIT_NOISE = {"LB","LBS","KG","KGS","KPA","PSI","BAR","KPA)","PSI)","KPA(","PSI("}
_CONTEXT_NOISE = {"MAX","LOAD","INFLATION","PRESSURE","WEIGHT","COLD"}
_NOISE = _UNIT_NOISE | _CONTEXT_NOISE | {
    "RADIAL","TUBELESS","OUTSIDE","SAFETY","WARNING","MADE","IN","SIDEWALL","TREAD",
    "POLYESTER","NYLON","STEEL","PLY","PLIES","TEMPERATURE","TRACTION","TREADWEAR",
    "RIM","RIMS","STANDARD","MOUNT","ONLY","APPROVED","FOR","CAUTION","EXPLOSION",
    "OWNER","MANUAL","VEHICLE","FOLLOW","DOT"
}

# OCR digit repair map
_OCR_DIGIT_MAP = str.maketrans({
    "O":"0","o":"0",
    "I":"1","i":"1","L":"1","l":"1","(":"1",")":"",
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
    Load plant metadata from (first match wins):
      env: PLANT_CODES_JSON
      /mnt/data/plant_codes.json
      data/plant_codes.json
    Formats:
      dict: {"16C": {"manufacturer": "...", "plant": "...", "country": "..."}}
      list: [{"code":"16C","manufacturer":"...","plant":"...","country":"..."}]
    """
    global _PLANT_MAP
    _PLANT_MAP.clear()
    for path in (os.environ.get("PLANT_CODES_JSON") or "",
                 "/mnt/data/plant_codes.json",
                 "data/plant_codes.json"):
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    code = str(k).upper().strip()
                    _PLANT_MAP[code] = {
                        "manufacturer": str(v.get("manufacturer","")).strip(),
                        "plant": str(v.get("plant","")).strip(),
                        "country": str(v.get("country","")).strip(),
                    }
            elif isinstance(data, list):
                for row in data:
                    code = str(row.get("code","")).upper().strip()
                    if not code: continue
                    _PLANT_MAP[code] = {
                        "manufacturer": str(row.get("manufacturer","")).strip(),
                        "plant": str(row.get("plant","")).strip(),
                        "country": str(row.get("country","")).strip(),
                    }
            if _PLANT_MAP:
                break
        except Exception:
            pass

def _lookup_plant(code: str) -> Dict[str, str]:
    return _PLANT_MAP.get((code or "").upper(), {})

_load_plant_codes()

# ===============================
# Utilities
# ===============================

_TOKEN_RE = re.compile(r"[A-Za-z0-9\(\)\[\]\-_/\.]+")

def _clean(tok: str, digits_mode: bool = False) -> str:
    t = tok.strip()
    t = re.sub(r"^[\(\[\{]+|[\)\]\}]+$", "", t)
    t = re.sub(r"[\/\.\-]+", "", t)
    t = t.upper()
    if digits_mode:
        t = t.translate(_OCR_DIGIT_MAP)
    return re.sub(r"[^A-Z0-9]", "", t)

def _tokenize_window(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)

def _digit_windows(token: str) -> List[str]:
    out: List[str] = []
    for d in re.findall(r"\d+", token):
        if len(d) == 4:
            out.append(d)
        elif len(d) > 4:
            for i in range(0, len(d)-3):
                out.append(d[i:i+4])
    return out

# ---------- line-aware tokenization ----------

@dataclass
class LineTokens:
    raw: List[List[str]]
    clean: List[List[str]]
    digits: List[List[str]]
    flat_raw: List[str]
    flat_clean: List[str]
    flat_digits: List[str]
    map_global_to_line: List[Tuple[int,int]]
    dot_lines: Dict[int,int]     # line_idx -> first DOT pos
    unit_lines: set              # line indices that look like unit/pressure/load lines

def _tokenize_lines(text: str) -> LineTokens:
    lines = text.splitlines()
    raw_per = [_tokenize_window(ln) for ln in lines]
    clean_per = [[_clean(t, False) for t in row if t.strip()] for row in raw_per]
    digits_per = [[_clean(t, True)  for t in row if t.strip()] for row in raw_per]

    flat_raw: List[str] = []
    flat_clean: List[str] = []
    flat_digits: List[str] = []
    map_gl: List[Tuple[int,int]] = []
    for li, row in enumerate(raw_per):
        for pi, tok in enumerate(row):
            ct = _clean(tok, False)
            dt = _clean(tok, True)
            if ct or re.search(r"\d", tok):
                flat_raw.append(tok)
                flat_clean.append(ct or tok)
                flat_digits.append(dt or tok)
                map_gl.append((li, pi))

    dot_lines: Dict[int,int] = {}
    for li, row in enumerate(clean_per):
        for pi, ct in enumerate(row):
            if ct == "DOT":
                dot_lines[li] = pi
                break

    unit_lines = set()
    for li, row in enumerate(clean_per):
        if li in dot_lines: continue
        if set(row) & (_UNIT_NOISE | _CONTEXT_NOISE):
            unit_lines.add(li)

    return LineTokens(
        raw=raw_per, clean=clean_per, digits=digits_per,
        flat_raw=flat_raw, flat_clean=flat_clean, flat_digits=flat_digits,
        map_global_to_line=map_gl, dot_lines=dot_lines, unit_lines=unit_lines
    )

# ===============================
# Date candidate scoring
# ===============================

@dataclass
class DateCandidate:
    token_idx: int
    wwyy: str
    week: Optional[int]
    year: Optional[int]
    score: float
    reasons: List[str]

def _windows_after_dot_chars(text: str, span: int = DATE_WINDOW_SPAN_CHARS) -> List[Tuple[int,int]]:
    spans: List[Tuple[int,int]] = []
    for m in re.finditer(r"(?i)\bDOT\b", text):
        start = m.end()
        spans.append((start, min(len(text), start + span)))
    return spans

def _week_ok(w: int) -> bool:
    return 1 <= w <= 53

def _year_penalty(yy: int) -> float:
    now_yy = int(datetime.now().strftime("%y"))
    return 0.5 if yy > now_yy + FUTURE_YEAR_SOFT_TOLERANCE else 0.0

def _end_of_dot_line_bonus(pos_in_line: int, line_len: int) -> float:
    if line_len <= 1: return 0.0
    frac = pos_in_line / (line_len - 1)
    return 0.8 * frac

def _score_date_candidates(text: str, L: LineTokens, dot_spans: List[Tuple[int,int]]) -> List[DateCandidate]:
    candidates: List[DateCandidate] = []

    # build offsets for "near DOT" check
    joined = " ".join(L.flat_digits)
    offs: List[Tuple[int,int]] = []
    pos = 0
    for t in L.flat_digits:
        offs.append((pos, pos + len(t)))
        pos += len(t) + 1

    def near_dot(idx: int) -> bool:
        if not offs or not dot_spans: return False
        ts, te = offs[idx]
        return any((ts >= ds and ts <= de) or (te >= ds and te <= de) for (ds, de) in dot_spans)

    size_like   = re.compile(r"\b\d{3}/\d{2}R\d{2}[A-Z0-9]?\b")
    load_like   = re.compile(r"\b\d{2,3}/\d{2,3}\b")
    wheel_inch  = re.compile(r"\b(17\.5|19\.5|22\.5|16|17|18|19|20|21|22|24)\b")
    has_slash   = re.compile(r"/")

    def penalize_size_load(raw_tok: str, clean_tok: str) -> bool:
        rt = raw_tok.upper()
        if size_like.search(rt) or load_like.search(rt): return True
        if has_slash.search(rt) or wheel_inch.search(rt): return True
        if rt.strip().endswith("R"): return True
        return False

    def is_bracketed(raw_tok: str, wwyy: str) -> bool:
        return bool(re.search(rf"[\(\[\{{]\s*{wwyy}\s*[\)\]\}}]", raw_tok))

    def line_len(li: int) -> int:
        return len(L.raw[li]) if 0 <= li < len(L.raw) else 0

    for i, (tc, td) in enumerate(zip(L.flat_clean, L.flat_digits)):
        raw_tok = L.flat_raw[i]
        line_idx, pos_in_line = L.map_global_to_line[i]
        on_dot_line = line_idx in L.dot_lines
        dot_pos = L.dot_lines.get(line_idx, -1)

        # exact 4-digit
        if re.fullmatch(r"\d{4}", td):
            wwyy = td
            wk, yy = int(wwyy[:2]), int(wwyy[2:])
            score, reasons = 0.0, []
            if on_dot_line:
                score += 4.8; reasons.append("on DOT line")
                if dot_pos >= 0 and pos_in_line > dot_pos:
                    score += 2.6; reasons.append("after DOT on line")
                score += _end_of_dot_line_bonus(pos_in_line, line_len(line_idx)); reasons.append("end-of-dot-line nudge")
            elif near_dot(i):
                score += 1.8; reasons.append("near DOT")
            if is_bracketed(raw_tok, wwyy):
                score += 0.8; reasons.append("bracketed")
            score += 0.8; reasons.append("token==WWYY")
            if _week_ok(wk):
                score += 1.5; reasons.append("valid week")
            else:
                reasons.append("week out of range")
            if (line_idx in L.unit_lines) and (not on_dot_line):
                score -= 2.0; reasons.append("unit line")
            if penalize_size_load(raw_tok, tc):
                score -= 2.7; reasons.append("size/load token")
            score -= _year_penalty(yy)
            candidates.append(DateCandidate(i, wwyy, wk if _week_ok(wk) else None, yy, score, reasons))

        # embedded windows
        for wwyy in _digit_windows(td):
            wk, yy = int(wwyy[:2]), int(wwyy[2:])
            score, reasons = 0.0, []
            if on_dot_line:
                score += 4.2; reasons.append("on DOT line")
                if dot_pos >= 0 and pos_in_line > dot_pos:
                    score += 2.3; reasons.append("after DOT on line")
                score += _end_of_dot_line_bonus(pos_in_line, line_len(line_idx)); reasons.append("end-of-dot-line nudge")
            elif near_dot(i):
                score += 1.5; reasons.append("near DOT")
            if is_bracketed(raw_tok, wwyy):
                score += 0.6; reasons.append("bracketed")
            score += 0.25; reasons.append("embedded WWYY")
            if _week_ok(wk):
                score += 1.5; reasons.append("valid week")
            else:
                reasons.append("week out of range")
            if (line_idx in L.unit_lines) and (not on_dot_line):
                score -= 2.0; reasons.append("unit line")
            if penalize_size_load(raw_tok, tc):
                score -= 2.7; reasons.append("size/load token")
            score -= _year_penalty(yy)
            candidates.append(DateCandidate(i, wwyy, wk if _week_ok(wk) else None, yy, score, reasons))

    candidates.sort(key=lambda c: (c.score, c.token_idx), reverse=True)
    return candidates

# ===============================
# Plant & serial extraction
# ===============================

def _looks_like_new_plant(code: str) -> bool:
    """Length 2–3, alnum, first char in 0/1/2 (new scheme)."""
    return (PLANT_MIN_LEN <= len(code) <= PLANT_MAX_LEN) and code.isalnum() and code[0] in "012"

def _find_plant_after_dot_lines(L: LineTokens) -> Tuple[str, int]:
    """
    Find a 2–3 char alnum plant immediately after DOT on the DOT line.
    Prefer new-scheme (leading 0/1/2). Returns (plant_code, global_token_index) or ("", -1).
    """
    best: Tuple[str, int, int] = ("", -1, 99)  # code, gidx, tier (0 best)
    for li, first_dot_pos in L.dot_lines.items():
        row_raw = L.raw[li]
        raw_clean = [_clean(t, False) for t in row_raw]
        try:
            p_dot = raw_clean.index("DOT")
        except ValueError:
            p_dot = first_dot_pos

        look = range(p_dot + 1, min(p_dot + 6, len(row_raw)))
        for pi in look:
            candL = _clean(row_raw[pi], False)
            candD = _clean(row_raw[pi], True)
            for cand in (candD, candL):
                if not cand: continue
                if _looks_like_new_plant(cand):
                    tier = 0
                elif PLANT_MIN_LEN <= len(cand) <= PLANT_MAX_LEN and cand.isalnum():
                    tier = 1
                else:
                    continue
                # map line-local to global
                for gi, (g_li, g_pi) in enumerate(L.map_global_to_line):
                    if g_li == li and g_pi == pi:
                        if tier < best[2] or (tier == best[2] and gi < best[1] or best[1] == -1):
                            best = (cand[:PLANT_MAX_LEN], gi, tier)
                        break
        if best[1] != -1:
            break
    return (best[0], best[1]) if best[1] != -1 else ("", -1)

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
    Returns:
    {
      "full": "DOT <PLANT?> <SERIAL?> <WWYY?>",
      "parts": {
        "Plant","PlantName","PlantCountry","PlantManufacturer",
        "Serial","Date","Week","Year"
      },
      "_debug": { "date_candidates":[...], "chosen_from_dot_line":bool,
                  "accepted":bool, "accept_reasons":[...], "had_dot":bool }
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
            "chosen_from_dot_line": False,
            "accepted": False,
            "accept_reasons": [],
            "had_dot": False
        }
    }
    if not text:
        return out

    L = _tokenize_lines(text)
    if not L.flat_raw:
        return out

    HAS_DOT = bool(re.search(r"(?i)\bDOT\b", text))
    out["_debug"]["had_dot"] = HAS_DOT

    # Try to get plant immediately; we want to return it even if no date is accepted.
    plant, plant_gidx = _find_plant_after_dot_lines(L)

    # Build a best-effort serial using tokens that appear before the date (if we find one),
    # otherwise use a short tail of tokens after plant/DOT.
    serial = ""
    serial_from_tail = ""
    if plant:
        # take a short tail after plant token in flattened clean tokens
        flat_before = L.flat_clean[:]
        if plant in flat_before:
            p = flat_before.index(plant)
            tail = flat_before[p+1:p+6]
            serial_from_tail = _concat_middle(tail, 0, len(tail))
    else:
        # if no plant, take a tiny tail after first DOT token on a DOT line
        if L.dot_lines:
            li0 = next(iter(L.dot_lines))
            dot_pos = L.dot_lines[li0]
            row = L.raw[li0]
            tail_clean = [_clean(t, False) for t in row[dot_pos+1:dot_pos+6]]
            serial_from_tail = _concat_middle(tail_clean, 0, len(tail_clean))

    # Date scoring + acceptance gate
    dot_spans = _windows_after_dot_chars(text, DATE_WINDOW_SPAN_CHARS)
    candidates = _score_date_candidates(text, L, dot_spans)
    out["_debug"]["date_candidates"] = [
        {"token_idx": c.token_idx, "wwyy": c.wwyy, "week": c.week, "year": c.year,
         "score": round(c.score, 3), "reasons": c.reasons}
        for c in candidates[:16]
    ]

    MIN_DATE_SCORE = 3.6            # with DOT present
    MIN_DATE_SCORE_NO_DOT = 5.2     # no DOT → stricter & must be bracketed

    def _prox_ok(c: DateCandidate) -> bool:
        rs = set(c.reasons)
        return ("on DOT line" in rs) or ("after DOT on line" in rs) or ("near DOT" in rs)

    def _bracketed(c: DateCandidate) -> bool:
        return any("bracketed" in r for r in c.reasons)

    accepted: List[DateCandidate] = []
    for c in candidates:
        if not c.week:
            continue
        if HAS_DOT:
            if _prox_ok(c) and c.score >= MIN_DATE_SCORE:
                accepted.append(c)
        else:
            if _bracketed(c) and c.score >= MIN_DATE_SCORE_NO_DOT:
                accepted.append(c)

    date_wwyy = ""
    if accepted:
        best = sorted(accepted, key=lambda x: (x.score, x.token_idx), reverse=True)[0]
        date_wwyy = best.wwyy
        li, _ = L.map_global_to_line[best.token_idx]
        out["_debug"]["chosen_from_dot_line"] = (li in L.dot_lines)
        out["_debug"]["accepted"] = True
        out["_debug"]["accept_reasons"] = best.reasons

        # build serial using tokens BEFORE the date in flattened sequence
        before_clean = L.flat_clean[:best.token_idx]
        if plant and plant in before_clean:
            p_idx = before_clean.index(plant)
            serial = _concat_middle(before_clean, p_idx + 1, len(before_clean))
        else:
            tail = before_clean[-3:] if len(before_clean) >= 3 else before_clean
            serial = _concat_middle(tail, 0, len(tail))
    else:
        # no accepted date: keep serial from short tail after plant/DOT
        serial = serial_from_tail

    # Human-readable full
    blocks: List[str] = []
    if HAS_DOT: blocks.append("DOT")
    if plant:   blocks.append(plant)
    if serial:  blocks.append(serial)
    if date_wwyy: blocks.append(date_wwyy)
    full_str = " ".join(blocks).strip()

    # Plant meta
    meta = _lookup_plant(plant) if plant else {}

    out["full"] = full_str
    out["parts"].update({
        "Plant": plant,
        "PlantName": meta.get("plant", ""),
        "PlantCountry": meta.get("country", ""),
        "PlantManufacturer": meta.get("manufacturer", ""),
        "Serial": serial,
        "Date": date_wwyy,
        "Week": date_wwyy[:2] if date_wwyy else "",
        "Year": date_wwyy[2:] if date_wwyy else "",
    })
    return out

# Optional: quick CLI
if __name__ == "__main__":
    import sys, pprint
    txt = sys.stdin.read() if not sys.argv[1:] else open(sys.argv[1], "r", encoding="utf-8").read()
    pprint.pp(parse_tin(txt))