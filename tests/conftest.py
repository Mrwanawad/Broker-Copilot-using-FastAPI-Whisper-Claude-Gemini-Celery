import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


@pytest.fixture(autouse=True)
def _settings(tmp_path, monkeypatch):
    monkeypatch.setenv("BC_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("BC_SUMMARIZER", "rules")
    monkeypatch.setenv("BC_REDIS_URL", "")
    monkeypatch.setenv("BC_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("BC_API_KEYS", "tester:test-key")
    monkeypatch.setenv("BC_TWILIO_AUTH_TOKEN", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")  # tests stay offline; Gemini tests stub the client
    from app.config import get_settings
    from app.db import session

    get_settings.cache_clear()
    session.get_engine.cache_clear()
    session._factory.cache_clear()
    yield
    get_settings.cache_clear()
    session.get_engine.cache_clear()
    session._factory.cache_clear()


CALL_TEXT = """broker: مساء الخير يا أستاذ أحمد، معاك منى من ناوي
client: أنا بدور على شقة في التجمع الخامس 3 اوض
client: ميزانيتي في حدود ٤ مليون ونص، وعايز تقسيط على 8 سنين
client: بس الأسعار غالية شوية
broker: هرتب لحضرتك معاينة يوم الخميس"""
