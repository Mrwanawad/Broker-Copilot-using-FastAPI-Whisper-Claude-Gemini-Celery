from datetime import date

import pytest

from app.text.dates import resolve
from app.text.gazetteer import find_areas, find_unit_types
from app.text.normalize import clean_display, normalize_search
from app.text.numbers import canonicalize_amounts, find_amounts, find_bedrooms, find_percents, find_years


@pytest.mark.parametrize(("text", "expected"), [
    ("ميزانيتي ٣ مليون ونص", [3_500_000]),
    ("عايز حاجة بمليونين", [2_000_000]),
    ("مقدم خمسمية ألف", [500_000]),
    ("budget 7.5M", [7_500_000]),
    ("تلاتة مليون و ٢٠٠ ألف", [3_200_000]),
    ("السعر 4,250,000 جنيه", [4_250_000]),
    ("مليون وربع", [1_250_000]),
    ("من 3 ل 4 مليون", [4_000_000]),  # "3" alone has no unit; range handled by the extractor
    ("مساحة 150 متر", []),
])
def test_find_amounts(text, expected):
    assert [a.value for a in find_amounts(text)] == expected


def test_percent_years_bedrooms():
    assert find_percents("10% مقدم") == [10.0]
    assert find_years("تقسيط على سبع سنين") == [7]
    assert find_bedrooms("شقة 3 اوض") == [3]
    assert find_bedrooms("شاليه أوضتين") == [2]


def test_canonicalize():
    assert canonicalize_amounts("ميزانيتي ٣ مليون ونص") == "ميزانيتي 3,500,000 EGP"


def test_normalization():
    assert clean_display("مَرْحَبًا ـــ ١٢٣") == "مرحبا 123"
    assert normalize_search("أحمد إسماعيل مدينة") == normalize_search("احمد اسماعيل مدينه")


def test_gazetteer_order_and_aliases():
    assert find_areas("بدور في التجمع الخامس أو الشيخ زايد") == ["New Cairo", "Sheikh Zayed"]
    assert find_unit_types("شقة 3 اوض أو تاون هاوس") == ["apartment", "townhouse"]


def test_relative_dates():
    thursday = date(2026, 11, 12)
    assert resolve("هكلمك بكرة", thursday) == date(2026, 11, 13)
    assert resolve("معاينة يوم السبت", thursday) == date(2026, 11, 14)
    assert resolve("يوم الخميس", thursday) == date(2026, 11, 19)  # same weekday means next week
    assert resolve("كمان 3 ايام", thursday) == date(2026, 11, 15)
    assert resolve("مفيش ميعاد", thursday) is None
