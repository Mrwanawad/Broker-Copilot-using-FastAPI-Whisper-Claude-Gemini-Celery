"""Arabic text normalization.

Two levels:
- `clean_display`: what the broker reads. Strips diacritics and tatweel, unifies digits. Keeps spelling.
- `normalize_search`: for matching/search only. Also folds alef/ya/teh-marbuta variants, so
  spelling drift between ASR runs lands on one key.

Uses camel-tools when installed (same rules, maintained tables); the fallback covers the same
characters so results do not depend on the optional dependency.
"""

import re

try:  # optional heavy dependency
    from camel_tools.utils.dediac import dediac_ar as _camel_dediac
    from camel_tools.utils.normalize import normalize_alef_ar as _camel_alef
    from camel_tools.utils.normalize import normalize_alef_maksura_ar as _camel_maksura
    from camel_tools.utils.normalize import normalize_teh_marbuta_ar as _camel_teh

    HAVE_CAMEL = True
except ImportError:  # pragma: no cover - depends on environment
    HAVE_CAMEL = False

DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
ARABIC_PUNCT = str.maketrans({"،": ",", "؛": ";", "؟": "?", "٪": "%", "٫": ".", "٬": ","})
ALEF = re.compile(r"[إأآٱ]")
SPACES = re.compile(r"\s+")


def unify_digits(text: str) -> str:
    return text.translate(ARABIC_INDIC).translate(ARABIC_PUNCT)


def dediac(text: str) -> str:
    if HAVE_CAMEL:
        return _camel_dediac(text)
    return DIACRITICS.sub("", text)


def clean_display(text: str) -> str:
    text = dediac(unify_digits(text)).replace(TATWEEL, "")
    return SPACES.sub(" ", text).strip()


def normalize_search(text: str) -> str:
    text = clean_display(text)
    if HAVE_CAMEL:
        text = _camel_teh(_camel_maksura(_camel_alef(text)))
    else:
        text = ALEF.sub("ا", text).replace("ى", "ي").replace("ة", "ه")
    return text.lower()
