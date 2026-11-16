"""Area and unit-type vocabulary: Egyptian-Arabic and English surface forms -> canonical names."""

from app.text.normalize import normalize_search

AREAS: dict[str, list[str]] = {
    "New Cairo": ["التجمع", "التجمع الخامس", "القاهرة الجديدة", "نيو كايرو", "new cairo", "5th settlement", "fifth settlement", "التسعين"],
    "Sheikh Zayed": ["زايد", "الشيخ زايد", "sheikh zayed", "zayed"],
    "6th of October": ["اكتوبر", "أكتوبر", "6 اكتوبر", "٦ أكتوبر", "october", "6th of october"],
    "New Capital": ["العاصمة", "العاصمة الإدارية", "العاصمة الادارية", "new capital", "administrative capital"],
    "North Coast": ["الساحل", "الساحل الشمالي", "الساحل الشمالى", "north coast", "sahel"],
    "Ain Sokhna": ["السخنة", "العين السخنة", "sokhna", "ain sokhna"],
    "Mostakbal City": ["المستقبل", "مدينة المستقبل", "mostakbal", "mostakbal city"],
    "Madinaty": ["مدينتي", "madinaty"],
    "El Shorouk": ["الشروق", "shorouk", "el shorouk"],
    "Maadi": ["المعادي", "maadi"],
    "Rehab": ["الرحاب", "rehab"],
}

UNIT_TYPES: dict[str, list[str]] = {
    "apartment": ["شقة", "شقه", "شقق", "apartment", "flat"],
    "villa": ["فيلا", "فيلل", "villa", "standalone", "ستاند الون"],
    "townhouse": ["تاون", "تاون هاوس", "townhouse", "town house"],
    "twin house": ["توين", "توين هاوس", "twin house", "twinhouse"],
    "duplex": ["دوبلكس", "duplex"],
    "penthouse": ["بنتهاوس", "penthouse"],
    "chalet": ["شاليه", "شليه", "شاليهات", "chalet"],  # شليه: common ASR spelling
    "studio": ["ستوديو", "استوديو", "studio"],
}

PURPOSES: dict[str, list[str]] = {
    "investment": ["استثمار", "استثماري", "investment", "اعيد بيع", "ريسيل"],
    "rental": ["ايجار", "إيجار", "اأجر", "rent", "rental"],
    "summer": ["مصيف", "الصيف", "summer"],
    "living": ["سكن", "اسكن", "هسكن", "اعيش", "living", "عيلة", "عيلتي"],
}

DELIVERY: dict[str, list[str]] = {
    "ready": ["استلام فوري", "جاهز", "متشطب وجاهز", "ready to move", "immediate delivery", "ready"],
    "within_1y": ["السنة الجاية", "خلال سنة", "next year"],
    "within_2y": ["خلال سنتين", "سنتين", "within two years"],
}


def _index(table: dict[str, list[str]]) -> list[tuple[str, str]]:
    pairs = [(normalize_search(alias), canonical) for canonical, aliases in table.items() for alias in aliases]
    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)  # longest alias wins


_AREA_INDEX = _index(AREAS)
_TYPE_INDEX = _index(UNIT_TYPES)
_PURPOSE_INDEX = _index(PURPOSES)
_DELIVERY_INDEX = _index(DELIVERY)


def _find(text: str, index: list[tuple[str, str]]) -> list[str]:
    norm = normalize_search(text)
    found: dict[str, int] = {}
    for alias, canonical in index:
        pos = norm.find(alias)
        if pos >= 0:
            found.setdefault(canonical, pos)
            norm = norm.replace(alias, " " * len(alias))  # keep offsets stable
    return sorted(found, key=found.__getitem__)  # in the order the speaker said them


def find_areas(text: str) -> list[str]:
    return _find(text, _AREA_INDEX)


def find_unit_types(text: str) -> list[str]:
    return _find(text, _TYPE_INDEX)


def find_purposes(text: str) -> list[str]:
    return _find(text, _PURPOSE_INDEX)


def find_delivery(text: str) -> list[str]:
    return _find(text, _DELIVERY_INDEX)
