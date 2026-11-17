"""LLM summarizer wiring (Claude primary, Gemini fallback), with SDK clients stubbed (no network)."""

from datetime import date
from types import SimpleNamespace

from app.channels.text_threads import parse_speaker_text
from app.domain import CallSummary, Evidence, Field_, Needs
from app.summarizer import claude, gemini, service
from tests.conftest import CALL_TEXT


class FakeMessages:
    def __init__(self, parsed, stop_reason="end_turn", error=None):
        self.parsed, self.stop_reason, self.error, self.kwargs = parsed, stop_reason, error, None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        usage = SimpleNamespace(input_tokens=1200, output_tokens=400)
        return SimpleNamespace(parsed_output=self.parsed, stop_reason=self.stop_reason, usage=usage,
                               model=kwargs["model"])


def _use_fake(monkeypatch, messages):
    monkeypatch.setenv("BC_SUMMARIZER", "claude")
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setattr(claude, "get_client", lambda: SimpleNamespace(messages=messages))


def test_claude_summary_is_verified(monkeypatch):
    parsed = CallSummary(
        headline="3BR apartment in New Cairo",
        needs=Needs(
            areas=["New Cairo"],
            budget_max_egp=Field_(value=4_500_000, confidence=0.95,
                                  evidence=[Evidence(segment=2, quote="ميزانيتي في حدود ٤ مليون ونص")]),
            bedrooms=Field_(value=4, confidence=0.9, evidence=[Evidence(segment=1, quote="4 اوض")]),  # invented
        ),
    )
    fake = FakeMessages(parsed)
    _use_fake(monkeypatch, fake)
    out = service.summarize(parse_speaker_text(CALL_TEXT), date(2026, 11, 12))
    assert fake.kwargs["output_format"] is CallSummary
    assert fake.kwargs["output_config"] == {"effort": "medium"}
    assert "[2]" in fake.kwargs["messages"][0]["content"]  # segment indexes are in the prompt
    s = out["summary"]
    assert s.summarizer.startswith("claude:")
    assert s.needs.budget_max_egp.confidence == 0.95
    assert out["flagged"] == ["needs.bedrooms"] and s.needs.bedrooms.confidence < 0.5
    assert out["usage"]["input_tokens"] == 1200


def test_claude_failure_falls_back_to_rules(monkeypatch):
    _use_fake(monkeypatch, FakeMessages(None, error=RuntimeError("network down")))
    out = service.summarize(parse_speaker_text(CALL_TEXT), date(2026, 11, 12))
    assert out["summary"].summarizer == "rules"
    assert "network down" in out["fallback_reason"]


def test_claude_refusal_falls_back(monkeypatch):
    _use_fake(monkeypatch, FakeMessages(CallSummary(headline="x"), stop_reason="refusal"))
    out = service.summarize(parse_speaker_text(CALL_TEXT), date(2026, 11, 12))
    assert out["summary"].summarizer == "rules" and "declined" in out["fallback_reason"]


class FakeModels:
    def __init__(self, summary):
        self.summary, self.kwargs = summary, None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        meta = SimpleNamespace(prompt_token_count=900, candidates_token_count=300, thoughts_token_count=100)
        cand = SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))
        return SimpleNamespace(text=self.summary.model_dump_json(), candidates=[cand], usage_metadata=meta,
                               response_id="r1")


def _use_fake_gemini(monkeypatch, summary):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    from app.config import get_settings

    get_settings.cache_clear()
    models = FakeModels(summary)
    monkeypatch.setattr(gemini, "get_client", lambda: SimpleNamespace(models=models))
    return models


def _gemini_summary():
    return CallSummary(headline="3BR in New Cairo", needs=Needs(
        budget_max_egp=Field_(value=4500000, confidence=0.95,
                              evidence=[Evidence(segment=2, quote="ميزانيتي في حدود ٤ مليون ونص")])))


def test_claude_failure_falls_back_to_gemini(monkeypatch):
    _use_fake(monkeypatch, FakeMessages(None, error=RuntimeError("network down")))
    models = _use_fake_gemini(monkeypatch, _gemini_summary())
    out = service.summarize(parse_speaker_text(CALL_TEXT), date(2026, 11, 12))
    assert out["summary"].summarizer.startswith("gemini:")
    assert "network down" in out["fallback_reason"]
    assert "[2]" in models.kwargs["contents"]
    assert out["usage"]["output_tokens"] == 400 and out["flagged"] == []


def test_auto_uses_gemini_without_anthropic_credentials(monkeypatch):
    monkeypatch.setenv("BC_SUMMARIZER", "auto")
    monkeypatch.setattr(claude, "claude_available", lambda: False)
    _use_fake_gemini(monkeypatch, _gemini_summary())
    out = service.summarize(parse_speaker_text(CALL_TEXT), date(2026, 11, 12))
    assert out["summary"].summarizer.startswith("gemini:") and out["fallback_reason"] is None


def test_gemini_eval_replay_runs_offline():
    from pathlib import Path

    from app.evaluation.run import run

    report = run(Path("data/eval/calls.jsonl"), "gemini", Path("data/eval/gemini_replay.jsonl"))
    assert report["synthetic"] and report["cases"] == 5
    assert all(r["summarizer"].startswith("gemini:") and r["fallback"] is None for r in report["rows"])


def test_empty_llm_needs_backfilled_from_rules(monkeypatch):
    monkeypatch.setenv("BC_SUMMARIZER", "gemini")
    _use_fake_gemini(monkeypatch, _gemini_summary())  # LLM gave only the budget
    out = service.summarize(parse_speaker_text(CALL_TEXT), date(2026, 11, 12))
    s = out["summary"]
    assert s.needs.budget_max_egp.confidence == 0.95  # LLM value kept
    assert s.needs.unit_type.value == "apartment" and s.needs.bedrooms.value == 3 and s.needs.areas == ["New Cairo"]
    assert {"unit_type", "bedrooms", "areas"} <= set(out["usage"]["backfilled_from_rules"])
    assert "budget_max_egp" not in out["usage"]["backfilled_from_rules"]
