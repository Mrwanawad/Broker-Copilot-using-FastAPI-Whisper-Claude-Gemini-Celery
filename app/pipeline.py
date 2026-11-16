"""End-to-end processing of one call/thread:

audio -> preprocess -> ASR -> diarization -> summary -> CRM diff -> unit matches -> follow-up draft
text  -> parse (WhatsApp export / speaker lines)  -^

Each stage is timed; the call row ends in REVIEW (broker approves CRM changes) or FAILED.
"""

import json
import logging
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app import storage
from app.config import get_settings
from app.crm.client import get_crm
from app.crm.mapping import propose_changes
from app.db.models import Call, CallStatus
from app.domain import Channel, Transcript
from app.matching.followup import draft_followup
from app.matching.matcher import get_matcher
from app.summarizer.service import summarize

log = logging.getLogger(__name__)

# USD per unit, for the per-call cost monitor
PRICE_PER_MTOK = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}
WHISPER_API_PER_MIN = 0.006


@contextmanager
def timed(timings: dict, stage: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = round(time.perf_counter() - t0, 3)


def transcribe_audio(path: Path, work_dir: Path, timings: dict) -> Transcript:
    from app.asr.engines import get_engine
    from app.audio.preprocess import prepare
    from app.diarization.diarize import diarize

    s = get_settings()
    with timed(timings, "preprocess"):
        prepared = prepare(path, work_dir, use_vad=s.use_vad, use_noise_reduction=s.use_noise_reduction)
    with timed(timings, "asr"):
        transcript = get_engine().transcribe(prepared.path, language=s.asr_language)
        transcript.duration_s = transcript.duration_s or prepared.duration_s
    with timed(timings, "diarization"):
        transcript = diarize(prepared.path, transcript)
    return transcript


def parse_text(call: Call) -> Transcript:
    from app.channels.text_threads import looks_like_whatsapp, parse_speaker_text, parse_whatsapp_export

    text = call.source_text or ""
    if looks_like_whatsapp(text):
        return parse_whatsapp_export(text, broker_names=[call.broker or "", "nawy"])
    return parse_speaker_text(text)


def estimate_cost(usage: dict, transcript: Transcript) -> dict:
    model = usage.get("model") or get_settings().claude_model
    price_in, price_out = next((v for k, v in PRICE_PER_MTOK.items() if model.startswith(k)), (4.0, 20.0))
    llm = usage.get("input_tokens", 0) / 1e6 * price_in + usage.get("output_tokens", 0) / 1e6 * price_out
    asr = transcript.duration_s / 60 * WHISPER_API_PER_MIN if transcript.asr_model == "openai:whisper-1" else 0.0
    return {"llm_usd": round(llm, 5), "asr_usd": round(asr, 5), "total_usd": round(llm + asr, 5)}


def process_call(db: Session, call_id: int) -> Call:
    call = db.get(Call, call_id)
    if call is None:
        raise KeyError(call_id)
    call.status = CallStatus.PROCESSING
    call.error = None
    db.commit()
    timings: dict = {}
    t_start = time.perf_counter()
    try:
        if call.channel == Channel.WHATSAPP_TEXT or (call.source_text and not call.source_path):
            with timed(timings, "parse"):
                transcript = parse_text(call)
        else:
            work = Path(get_settings().upload_dir) / f"call_{call.id}"
            with timed(timings, "fetch"):
                audio = storage.fetch(call.source_path, work)
            transcript = transcribe_audio(audio, work, timings)
        if not transcript.segments:
            raise ValueError("empty transcript: no speech or no parseable messages")

        crm = get_crm(db)
        lead = crm.get_lead(call.lead_id) if call.lead_id else None
        lead = lead or {}
        lead_context = json.dumps({k: v for k, v in lead.items() if v not in (None, [], "")}, ensure_ascii=False)
        call_date = (call.created_at or datetime.now(UTC)).date()
        with timed(timings, "summarize"):
            result = summarize(transcript, call_date, lead_context if lead else "")
        summary = result["summary"]
        with timed(timings, "crm_diff"):
            changes = propose_changes(summary, lead, get_settings().crm_confidence_threshold)
        with timed(timings, "match"):
            matches = get_matcher().match(summary.needs, top_k=3)
        followup = draft_followup(summary, matches, lead.get("name"))

        call.transcript = transcript.model_dump(mode="json")
        call.summary = summary.model_dump(mode="json")
        call.crm_changes = [c.model_dump(mode="json") for c in changes]
        call.flagged_fields = result["flagged"]
        call.matches = [m.model_dump(mode="json") for m in matches]
        call.followup_draft = followup
        call.duration_s = transcript.duration_s
        usage = dict(result["usage"])
        usage.update(estimate_cost(usage, transcript))
        if result["fallback_reason"]:
            usage["fallback_reason"] = result["fallback_reason"]
        call.usage = usage
        call.status = CallStatus.REVIEW
    except Exception as exc:
        log.exception("call %s failed", call_id)
        call.status = CallStatus.FAILED
        call.error = f"{type(exc).__name__}: {exc}"[:2000]
    call.timings = timings
    call.processing_s = round(time.perf_counter() - t_start, 3)
    call.processed_at = datetime.now(UTC)
    db.commit()
    return call
