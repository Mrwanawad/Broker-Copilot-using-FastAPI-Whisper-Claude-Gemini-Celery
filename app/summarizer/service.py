"""Summarizer entry point: pick Claude or rules, then verify evidence and normalize values."""

import logging
import re
from datetime import date
from difflib import SequenceMatcher

from app.config import get_settings
from app.domain import CallSummary, Evidence, Field_, Transcript
from app.summarizer import rules
from app.summarizer.playbook import SUGGESTED_RESPONSES
from app.text.normalize import normalize_search

log = logging.getLogger(__name__)

UNSUPPORTED_PENALTY = 0.4  # confidence multiplier when a cited quote is not in the transcript


def _quote_supported(ev: Evidence, transcript: Transcript) -> bool:
    quote = normalize_search(ev.quote)
    if not quote:
        return False
    segs = transcript.segments
    candidates = [segs[ev.segment]] if ev.segment is not None and 0 <= ev.segment < len(segs) else segs
    numbers = set(re.findall(r"\d+", quote))
    for seg in candidates:
        text = normalize_search(seg.text)
        if quote in text:
            return True
        # near-verbatim (model fixed a spelling or dropped a word): long quotes only, and every number
        # in the quote must be in the segment, since a changed number is exactly the error to catch
        if len(quote) < 12 or not numbers <= set(re.findall(r"\d+", text)):
            continue
        if SequenceMatcher(None, quote, text).find_longest_match(0, len(quote), 0, len(text)).size >= 0.85 * len(quote):
            return True
    return False


def verify_evidence(summary: CallSummary, transcript: Transcript) -> list[str]:
    """Cut confidence of fields whose quotes are not found. Returns the names of flagged fields."""
    flagged: list[str] = []

    def check(name: str, field: Field_) -> None:
        if field.value is None:
            return
        if not field.evidence:
            field.confidence = min(field.confidence, 0.5)
            flagged.append(name)
            return
        if not all(_quote_supported(e, transcript) for e in field.evidence):
            field.confidence = round(field.confidence * UNSUPPORTED_PENALTY, 3)
            flagged.append(name)

    for name in ("unit_type", "bedrooms", "budget_min_egp", "budget_max_egp", "down_payment_egp",
                 "payment_plan", "installment_years", "delivery", "purpose"):
        check(f"needs.{name}", getattr(summary.needs, name))
    check("lead_stage", summary.lead_stage)
    if summary.needs.areas and summary.needs.area_evidence and not all(
        _quote_supported(e, transcript) for e in summary.needs.area_evidence
    ):
        flagged.append("needs.areas")
    for obj in summary.objections:
        if not obj.suggested_response:
            obj.suggested_response = SUGGESTED_RESPONSES.get(obj.category, SUGGESTED_RESPONSES["other"])
    return flagged


def _providers() -> list[str]:
    """LLM providers to try, in order. Empty list: rules only."""
    mode = get_settings().summarizer
    if mode == "rules":
        return []
    from app.summarizer.claude import claude_available
    from app.summarizer.gemini import gemini_available

    if mode == "claude":
        return ["claude"] + (["gemini"] if gemini_available() else [])
    if mode == "gemini":
        return ["gemini"]
    return [name for name, ok in (("claude", claude_available()), ("gemini", gemini_available())) if ok]


def summarize(transcript: Transcript, call_date: date | None = None, lead_context: str = "") -> dict:
    """Returns {"summary": CallSummary, "flagged": [...], "usage": {...}, "fallback_reason": str|None}."""
    call_date = call_date or date.today()
    usage: dict = {}
    failures: list[str] = []
    summary: CallSummary | None = None
    for name in _providers():
        if name == "claude":
            from app.summarizer import claude as provider
        else:
            from app.summarizer import gemini as provider
        try:
            summary, usage = provider.summarize(transcript, call_date, lead_context)
            break
        except Exception as exc:  # network, refusal, schema: the next provider (or rules) keeps the call processed
            log.exception("%s summarizer failed; trying next", name)
            failures.append(f"{name}: {type(exc).__name__}: {exc}"[:300])
    if summary is None:
        summary = rules.extract(transcript, call_date)
    flagged = verify_evidence(summary, transcript)
    fallback_reason = " | ".join(failures)[:600] or None
    return {"summary": summary, "flagged": flagged, "usage": usage, "fallback_reason": fallback_reason}
