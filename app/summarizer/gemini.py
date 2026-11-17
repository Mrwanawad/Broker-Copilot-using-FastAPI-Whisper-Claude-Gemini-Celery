"""Gemini structured-output summarizer: fallback when no Anthropic credentials, or when Claude fails.

Same prompt and the same `CallSummary` schema as the Claude path, so `verify_evidence` treats both alike.
"""

import logging
import time
from datetime import date
from functools import lru_cache

from app.config import get_settings
from app.domain import CallSummary, Transcript
from app.summarizer.prompt import SYSTEM_PROMPT, build_user_prompt

log = logging.getLogger(__name__)

RETRY_CODES = {429, 500, 503}  # rate limit (free tier: ~5 RPM) and "model overloaded"
RETRY_WAITS_S = (5, 15, 30)


@lru_cache
def get_client():
    from google import genai

    return genai.Client(api_key=get_settings().google_api_key)


def gemini_available() -> bool:
    if not get_settings().google_api_key:
        return False
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False
    return True


def _generate(**kwargs):
    for wait in (*RETRY_WAITS_S, None):
        try:
            return get_client().models.generate_content(**kwargs)
        except Exception as exc:
            if wait is None or getattr(exc, "code", None) not in RETRY_CODES:
                raise
            log.warning("Gemini %s, retrying in %ss", exc.code, wait)
            time.sleep(wait)


def summarize(transcript: Transcript, call_date: date, lead_context: str = "") -> tuple[CallSummary, dict]:
    from google.genai import types

    settings = get_settings()
    response = _generate(
        model=settings.gemini_model,
        contents=build_user_prompt(transcript, call_date, lead_context),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=CallSummary.model_json_schema(),
        ),
    )
    candidate = response.candidates[0] if response.candidates else None
    finish = candidate.finish_reason.name if candidate and candidate.finish_reason else None
    if not response.text or finish not in (None, "STOP"):
        raise RuntimeError(f"Gemini summary incomplete (finish_reason={finish})")
    summary = CallSummary.model_validate_json(response.text)
    meta = response.usage_metadata
    usage = {
        "input_tokens": meta.prompt_token_count if meta else None,
        "output_tokens": ((meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)) if meta else None,
        "model": settings.gemini_model,
        "request_id": response.response_id,
    }
    summary.summarizer = f"gemini:{settings.gemini_model}"
    return summary, usage
