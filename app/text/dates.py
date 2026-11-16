"""Relative date phrases in Egyptian Arabic/English -> concrete dates, anchored on the call date."""

import re
from datetime import date, timedelta

from app.text.normalize import normalize_search

# Python weekday(): Monday=0 ... Sunday=6
WEEKDAYS = {
    "الاتنين": 0, "الاثنين": 0, "monday": 0,
    "التلات": 1, "الثلاثاء": 1, "التلاتاء": 1, "tuesday": 1,
    "الاربع": 2, "الاربعاء": 2, "wednesday": 2,
    "الخميس": 3, "thursday": 3,
    "الجمعه": 4, "الجمعة": 4, "friday": 4,
    "السبت": 5, "saturday": 5,
    "الحد": 6, "الاحد": 6, "sunday": 6,
}
RELATIVE = [
    ("بعد بكره", 2), ("بعد بكرة", 2), ("day after tomorrow", 2),
    ("بكره", 1), ("بكرة", 1), ("tomorrow", 1),
    ("النهارده", 0), ("النهاردة", 0), ("today", 0),
    ("الاسبوع الجاي", 7), ("الأسبوع الجاي", 7), ("next week", 7),
    ("الشهر الجاي", 30), ("next month", 30),
]
IN_DAYS = re.compile(r"(?:كمان|بعد|خلال|in)\s*(\d{1,2})\s*(?:ايام|أيام|يوم|days?)")


def resolve(text: str, anchor: date) -> date | None:
    norm = normalize_search(text)
    for phrase, days in RELATIVE:
        if normalize_search(phrase) in norm:
            return anchor + timedelta(days=days)
    m = IN_DAYS.search(norm)
    if m:
        return anchor + timedelta(days=int(m.group(1)))
    for name, wd in WEEKDAYS.items():
        if re.search(rf"(?:^|\s|يوم\s?){re.escape(normalize_search(name))}(?:\s|$|[,.?!])", norm):
            delta = (wd - anchor.weekday()) % 7 or 7
            return anchor + timedelta(days=delta)
    return None
