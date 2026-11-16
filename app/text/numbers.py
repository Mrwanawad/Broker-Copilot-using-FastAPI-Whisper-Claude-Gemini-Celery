"""Egyptian-Arabic money, percentage and duration parsing.

Clients say prices like "٣ مليون ونص", "مليونين", "خمسمية ألف", "7.5M", "تلاتة مليون و ٢٠٠ ألف".
ASR writes them inconsistently. This module finds those mentions and returns canonical integers,
and can rewrite the text so the summarizer and the search index see one form ("3,500,000 EGP").
"""

import re
from dataclasses import dataclass

from app.text.normalize import clean_display

WORD_NUMBERS: dict[str, float] = {
    "واحد": 1, "واحده": 1, "واحدة": 1,
    "اتنين": 2, "اثنين": 2, "إتنين": 2,
    "تلاته": 3, "تلاتة": 3, "ثلاثة": 3, "ثلاثه": 3, "تلات": 3, "ثلاث": 3, "تلاة": 3,
    "اربعه": 4, "اربعة": 4, "أربعة": 4, "أربعه": 4, "اربع": 4,
    "خمسه": 5, "خمسة": 5, "خمس": 5,
    "سته": 6, "ستة": 6, "ست": 6,
    "سبعه": 7, "سبعة": 7, "سبع": 7,
    "تمانيه": 8, "تمانية": 8, "ثمانية": 8, "ثمانيه": 8, "ثمان": 8, "ثماني": 8, "تمن": 8, "تمان": 8,
    "تسعه": 9, "تسعة": 9, "تسع": 9,
    "عشره": 10, "عشرة": 10, "عشر": 10,
    "حداشر": 11, "اتناشر": 12, "تلتاشر": 13, "اربعتاشر": 14, "خمستاشر": 15,
    "عشرين": 20, "تلاتين": 30, "ثلاثين": 30, "اربعين": 40, "خمسين": 50,
    "ستين": 60, "سبعين": 70, "تمانين": 80, "تسعين": 90,
    "ميه": 100, "مية": 100, "مائة": 100, "مئة": 100,
    "ميتين": 200, "تلتمية": 300, "تلاتمية": 300, "ربعمية": 400, "اربعمية": 400,
    "خمسمية": 500, "خمسميه": 500, "ستمية": 600, "سبعمية": 700, "تمنمية": 800, "تسعمية": 900,
}
DUAL_UNITS = {"مليونين": 2_000_000, "الفين": 2_000, "ألفين": 2_000}
UNITS: dict[str, int] = {
    "مليار": 1_000_000_000, "billion": 1_000_000_000, "bn": 1_000_000_000,
    "مليون": 1_000_000, "ملايين": 1_000_000, "million": 1_000_000, "mil": 1_000_000, "m": 1_000_000,
    "الف": 1_000, "ألف": 1_000, "آلاف": 1_000, "الاف": 1_000, "k": 1_000, "thousand": 1_000,
}
FRACTIONS = {"نص": 0.5, "نصف": 0.5, "ربع": 0.25, "تلت": 1 / 3}

_num = r"\d+(?:[.,]\d+)?"
_word = "|".join(sorted((re.escape(w) for w in WORD_NUMBERS), key=len, reverse=True))
_unit = "|".join(sorted((re.escape(u) for u in UNITS), key=len, reverse=True))
_dual = "|".join(DUAL_UNITS)
_frac = "|".join(FRACTIONS)

# One term: [number] unit [and a fraction], e.g. "3 مليون ونص", "مليونين", "500 ألف", "7.5M".
# The lookahead after the unit stops "m" matching inside "metres" or Arabic words.
TERM_RE = re.compile(
    rf"(?:(?P<dual>{_dual})|(?:(?P<n>{_num})|(?P<w>{_word}))?\s*(?P<u>{_unit})(?![a-zA-Z؀-ۿ]))"
    rf"(?:\s*و\s*(?P<frac>{_frac})(?![؀-ۿ]))?",
    re.IGNORECASE,
)
PERCENT_RE = re.compile(
    rf"(?P<n>{_num}|{_word})\s*(?:%|في\s*الم[يئ][هة]|بالم[يئ][هة]|percent)", re.IGNORECASE
)
YEARS_RE = re.compile(rf"(?P<n>{_num}|{_word})\s*(?:سنين|سنوات|سنه|سنة|years?|yrs?)", re.IGNORECASE)
BEDROOMS_RE = re.compile(
    rf"(?P<n>\d|{_word})\s*(?:اوض|أوض|غرف|غرفة|غرفه|اوضة|اوضه|bedrooms?|beds?|br)", re.IGNORECASE
)
BARE_NUMBER_RE = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d{6,})(?![\d.,])")


@dataclass(frozen=True)
class Amount:
    value: int
    start: int
    end: int
    raw: str


def _to_float(token: str) -> float:
    if token in WORD_NUMBERS:
        return WORD_NUMBERS[token]
    return float(token.replace(",", "."))


def _term_value(m: re.Match) -> tuple[float, float]:
    """(value, unit) of one matched term."""
    if m.group("dual"):
        value = float(DUAL_UNITS[m.group("dual")])
        unit = value / 2
    else:
        unit = float(UNITS[m.group("u").lower()])
        base = 1.0
        if m.group("n"):
            base = _to_float(m.group("n"))
        elif m.group("w"):
            base = WORD_NUMBERS[m.group("w")]
        value = base * unit
    if m.group("frac"):
        value += FRACTIONS[m.group("frac")] * unit
    return value, unit


def find_amounts(text: str, min_value: int = 10_000) -> list[Amount]:
    """Money amounts in a sentence. Values under `min_value` are dropped (sizes, counts)."""
    text = clean_display(text)
    amounts: list[Amount] = []
    pos = 0
    while pos < len(text):
        m = TERM_RE.search(text, pos)
        if not m:
            break
        total, unit = _term_value(m)
        start, end = m.start(), m.end()
        # absorb chained terms with a smaller unit: "3 مليون و 200 ألف"
        while True:
            joiner = re.match(r"\s*و\s*", text[end:])
            if not joiner:
                break
            m2 = TERM_RE.match(text, end + joiner.end())
            if not m2:
                break
            add, add_unit = _term_value(m2)
            if add_unit >= unit:
                break
            total += add
            end = m2.end()
        raw = text[start:end].strip()
        if total >= min_value:
            amounts.append(Amount(int(round(total)), start, end, raw))
        pos = max(end, start + 1)
    for m in BARE_NUMBER_RE.finditer(text):
        if any(a.start <= m.start() < a.end for a in amounts):
            continue
        value = int(m.group(1).replace(",", ""))
        if value >= min_value:
            amounts.append(Amount(value, m.start(), m.end(), m.group(1)))
    return sorted(amounts, key=lambda a: a.start)


def find_percents(text: str) -> list[float]:
    return [_to_float(m.group("n")) for m in PERCENT_RE.finditer(clean_display(text))]


def find_years(text: str) -> list[int]:
    out = []
    for m in YEARS_RE.finditer(clean_display(text)):
        value = _to_float(m.group("n"))
        if 1 <= value <= 15:
            out.append(int(value))
    return out


BEDROOM_DUALS = re.compile(r"(?:[اأ]وض[تط]ين|غرفتين|اوضتان|غرفتان)")  # اوضطين: ASR spelling


def find_bedrooms(text: str) -> list[int]:
    text = clean_display(text)
    out = [2] if BEDROOM_DUALS.search(text) else []
    for m in BEDROOMS_RE.finditer(text):
        value = _to_float(m.group("n"))
        if 1 <= value <= 7:
            out.append(int(value))
    return out


def canonicalize_amounts(text: str) -> str:
    """Rewrite spoken amounts as '3,500,000 EGP' (keeps the rest of the text untouched)."""
    text = clean_display(text)
    out, last = [], 0
    for a in find_amounts(text):
        out.append(text[last : a.start])
        out.append(f"{a.value:,} EGP")
        last = a.end
    out.append(text[last:])
    return "".join(out)


def format_egp(value: int | float | None) -> str:
    if value is None:
        return "-"
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}".rstrip("0").rstrip(".") + "M EGP"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K EGP"
    return f"{value:.0f} EGP"
