from fastapi.testclient import TestClient

from app.api.security import twilio_signature_ok
from tests.conftest import CALL_TEXT

H = {"X-API-Key": "test-key"}


def client() -> TestClient:
    from app.main import app

    return TestClient(app)


def test_auth_required():
    with client() as c:
        assert c.get("/api/leads").status_code == 401
        assert c.get("/health").status_code == 200


def test_text_call_review_and_approve():
    with client() as c:
        res = c.post("/api/calls/text", json={"lead_id": "L-1002", "text": CALL_TEXT, "channel": "call"}, headers=H)
        assert res.status_code == 202, res.text
        call = res.json()
        assert call["status"] == "review", call["error"]  # eager queue: processed inline
        fields = {c["field"]: c for c in call["crm_changes"]}
        assert fields["budget_max"]["status"] == "needs_review"  # lead L-1002 already has 6M
        assert fields["budget_max"]["new"] == 4_500_000
        assert 1 <= len(call["matches"]) <= 3 and all(m["unit_type"] == "apartment" for m in call["matches"])
        assert call["timings"]["summarize"] >= 0

        res = c.post(f"/api/calls/{call['id']}/approve", headers=H,
                     json={"approved_fields": ["budget_max"], "overrides": {"bedrooms": 2}})
        assert res.status_code == 200, res.text
        assert res.json()["approved_by"] == "tester"
        lead = c.get("/api/leads/L-1002", headers=H).json()
        assert lead["budget_max"] == 4_500_000
        assert lead["bedrooms"] == 2
        assert "New Cairo" in lead["preferred_areas"] and "Sheikh Zayed" in lead["preferred_areas"]

        assert c.post(f"/api/calls/{call['id']}/approve", headers=H, json={}).status_code == 409
        m = c.get("/api/metrics", headers=H).json()
        assert m["calls_by_status"]["approved"] == 1
        assert m["correction_rate_per_field"]["bedrooms"] == 1.0
        assert m["correction_rate_per_field"]["budget_max"] == 0.0


def test_audio_requires_consent():
    with client() as c:
        res = c.post("/api/calls/audio", headers=H, data={"consent": "false"},
                     files={"file": ("a.wav", b"RIFF", "audio/wav")})
        assert res.status_code == 422


def test_whatsapp_webhook_text_thread():
    with client() as c:
        assert c.get("/webhooks/whatsapp", params={"hub.mode": "subscribe", "hub.verify_token": "change-me",
                                                   "hub.challenge": "42"}).text == "42"
        msg = lambda body: {"entry": [{"changes": [{"value": {"messages": [  # noqa: E731
            {"from": "201000000001", "id": "x", "type": "text", "text": {"body": body}}]}}]}]}
        first = c.post("/webhooks/whatsapp", json=msg("عايز شقة في زايد")).json()["calls"][0]
        second = c.post("/webhooks/whatsapp", json=msg("ميزانيتي 5 مليون")).json()["calls"][0]
        assert first == second
        call = c.post(f"/api/calls/{first}/process", headers=H).json()
        assert call["lead_id"] == "L-1001"
        assert call["summary"]["needs"]["budget_max_egp"]["value"] == 5_000_000


def test_twilio_signature():
    import base64
    import hashlib
    import hmac

    url, params, token = "https://x.test/webhooks/twilio/recording", {"CallSid": "CA1", "A": "b"}, "secret"
    sig = base64.b64encode(hmac.new(token.encode(), (url + "Ab" + "CallSidCA1").encode(), hashlib.sha1).digest()).decode()
    assert twilio_signature_ok(url, params, sig, token)
    assert not twilio_signature_ok(url, params, sig, "other")
