"""Claude structured-output summarizer.

The transcript goes in with segment indexes; the response is parsed straight into `CallSummary`
(Pydantic) via `messages.parse`. Every value must cite the segment and a verbatim quote, and
`verify_evidence` checks those quotes against the transcript afterwards: a value whose quote is not
found has its confidence cut, so it falls below the CRM threshold and goes to the broker for review.
"""

import logging
import os
from datetime import date
from functools import lru_cache
from pathlib import Path

import anthropic

from app.config import get_settings
from app.domain import CallSummary, Transcript
from app.summarizer.prompt import SYSTEM_PROMPT, build_user_prompt

log = logging.getLogger(__name__)


@lru_cache
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(max_retries=3)


def claude_available() -> bool:
    """Credentials present: API key, auth token, or an `ant auth login` profile."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


def summarize(transcript: Transcript, call_date: date, lead_context: str = "") -> tuple[CallSummary, dict]:
    settings = get_settings()
    user = build_user_prompt(transcript, call_date, lead_context)
    response = get_client().messages.parse(
        model=settings.claude_model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        output_config={"effort": settings.claude_effort},
        messages=[{"role": "user", "content": user}],
        output_format=CallSummary,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to summarize this conversation")
    if response.stop_reason == "max_tokens" or response.parsed_output is None:
        raise RuntimeError(f"Claude summary incomplete (stop_reason={response.stop_reason})")
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "model": response.model,
        "request_id": getattr(response, "_request_id", None),
    }
    summary = response.parsed_output
    summary.summarizer = f"claude:{settings.claude_model}"
    return summary, usage
