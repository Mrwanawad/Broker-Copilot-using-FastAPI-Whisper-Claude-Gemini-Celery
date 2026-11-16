import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import PlainTextResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import storage
from app.api.schemas import ApproveIn, CallBrief, CallOut, MatchIn, TextCallIn
from app.api.security import require_api_key, verify_twilio
from app.config import get_settings
from app.crm.client import get_crm
from app.crm.mapping import apply_changes
from app.db.models import Call, CallStatus, FieldCorrection, Lead
from app.db.session import get_db
from app.domain import Channel, CRMChange, FieldStatus
from app.jobs.worker import enqueue
from app.matching.matcher import get_matcher

log = logging.getLogger(__name__)

api = APIRouter(dependencies=[Depends(require_api_key)])
hooks = APIRouter(prefix="/webhooks", tags=["webhooks"])

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".webm", ".aac", ".flac", ".amr"}


def _save_upload(data: bytes, suffix: str) -> str:
    s = get_settings()
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"file larger than {s.max_upload_mb} MB")
    return storage.put(data, suffix)


def _lead_by_phone(db: Session, phone: str | None) -> str | None:
    if not phone:
        return None
    digits = "".join(ch for ch in phone if ch.isdigit())[-10:]
    for lead in db.scalars(select(Lead)):
        if lead.phone and "".join(ch for ch in lead.phone if ch.isdigit()).endswith(digits):
            return lead.lead_id
    return None


# ---------------- calls ----------------


@api.post("/calls/audio", response_model=CallOut, status_code=202, tags=["calls"])
async def upload_audio(
    file: UploadFile = File(...),
    lead_id: str | None = Form(None),
    channel: Channel = Form(Channel.UPLOAD),
    consent: bool = Form(False),
    broker: str | None = Form(None),
    db: Session = Depends(get_db),
):
    if not consent:
        raise HTTPException(422, "recording consent is required before a call is processed")
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in AUDIO_EXT:
        raise HTTPException(415, f"unsupported audio type {suffix}")
    if lead_id and db.get(Lead, lead_id) is None:
        raise HTTPException(404, f"lead {lead_id} not found")
    path = _save_upload(await file.read(), suffix)
    call = Call(lead_id=lead_id, channel=channel, source_path=path, consent=True, broker=broker)
    db.add(call)
    db.commit()
    enqueue(call.id)
    db.refresh(call)
    return call


@api.post("/calls/text", response_model=CallOut, status_code=202, tags=["calls"])
def submit_text(body: TextCallIn, db: Session = Depends(get_db)):
    if body.lead_id and db.get(Lead, body.lead_id) is None:
        raise HTTPException(404, f"lead {body.lead_id} not found")
    call = Call(lead_id=body.lead_id, channel=body.channel, source_text=body.text, consent=body.consent,
                broker=body.broker)
    db.add(call)
    db.commit()
    enqueue(call.id)
    db.refresh(call)
    return call


@api.get("/calls", response_model=list[CallBrief], tags=["calls"])
def list_calls(status: str | None = None, lead_id: str | None = None, limit: int = Query(50, le=500),
               db: Session = Depends(get_db)):
    q = select(Call).order_by(Call.id.desc()).limit(limit)
    if status:
        q = q.where(Call.status == status)
    if lead_id:
        q = q.where(Call.lead_id == lead_id)
    return [CallBrief(id=c.id, lead_id=c.lead_id, channel=c.channel, status=c.status, created_at=c.created_at,
                      headline=(c.summary or {}).get("headline")) for c in db.scalars(q)]


@api.get("/calls/{call_id}", response_model=CallOut, tags=["calls"])
def get_call(call_id: int, db: Session = Depends(get_db)):
    call = db.get(Call, call_id)
    if call is None:
        raise HTTPException(404, "call not found")
    return call


@api.post("/calls/{call_id}/process", response_model=CallOut, status_code=202, tags=["calls"])
def reprocess(call_id: int, db: Session = Depends(get_db)):
    call = db.get(Call, call_id)
    if call is None:
        raise HTTPException(404, "call not found")
    if call.status == CallStatus.APPROVED:
        raise HTTPException(409, "call already approved")
    call.status = CallStatus.QUEUED
    db.commit()
    enqueue(call.id)
    db.refresh(call)
    return call


@api.post("/calls/{call_id}/approve", response_model=CallOut, tags=["calls"])
def approve(call_id: int, body: ApproveIn, actor: str = Depends(require_api_key), db: Session = Depends(get_db)):
    """Broker review: writes auto_apply fields (minus rejected), approved needs_review fields and corrections
    to the CRM, and logs every reviewed field for the correction-rate monitor."""
    call = db.get(Call, call_id)
    if call is None:
        raise HTTPException(404, "call not found")
    if call.status != CallStatus.REVIEW:
        raise HTTPException(409, f"call is {call.status}, not waiting for review")
    if not call.lead_id:
        raise HTTPException(422, "call is not linked to a lead")
    changes = [CRMChange(**c) for c in call.crm_changes or []]
    rejected = set(body.rejected_fields)
    kept = [c for c in changes if c.field not in rejected]
    crm = get_crm(db)
    lead = crm.get_lead(call.lead_id) or {}
    updated = apply_changes(lead, kept, set(body.approved_fields), body.overrides)
    diff = {k: v for k, v in updated.items() if lead.get(k) != v}
    crm.update_lead(call.lead_id, diff)

    proposed = {c.field for c in changes}
    for field, value in body.overrides.items():  # broker filled a field the copilot did not propose
        if field not in proposed:
            db.add(FieldCorrection(call_id=call.id, field=field, proposed=None, final=str(value),
                                   confidence=0.0, action="corrected"))
    for c in changes:
        if c.status == FieldStatus.UNCHANGED and c.field not in body.overrides:
            continue
        if c.field in body.overrides:
            action, final = "corrected", body.overrides[c.field]
        elif c.field in rejected or (c.status == FieldStatus.NEEDS_REVIEW and c.field not in body.approved_fields):
            action, final = "rejected", None
        elif c.status == FieldStatus.NEEDS_REVIEW:
            action, final = "accepted", c.new
        else:
            action, final = "auto", c.new
        db.add(FieldCorrection(call_id=call.id, field=c.field, proposed=str(c.new), confidence=c.confidence,
                               final=None if final is None else str(final), action=action))
    call.status = CallStatus.APPROVED
    call.approved_by = actor
    call.approved_at = datetime.now(UTC)
    db.commit()
    db.refresh(call)
    return call


# ---------------- leads, matching, metrics ----------------


@api.get("/leads", tags=["crm"])
def list_leads(db: Session = Depends(get_db)):
    return [lead.as_dict() for lead in db.scalars(select(Lead).order_by(Lead.lead_id))]


@api.get("/leads/{lead_id}", tags=["crm"])
def get_lead(lead_id: str, db: Session = Depends(get_db)):
    lead = db.get(Lead, lead_id)
    if lead is None:
        raise HTTPException(404, "lead not found")
    calls = db.scalars(select(Call).where(Call.lead_id == lead_id).order_by(Call.id.desc()).limit(20))
    return {**lead.as_dict(), "calls": [{"id": c.id, "status": c.status, "created_at": c.created_at,
                                         "headline": (c.summary or {}).get("headline")} for c in calls]}


@api.post("/match", tags=["matching"])
def match_units(body: MatchIn):
    return [m.model_dump() for m in get_matcher().match(body.needs, body.query, body.top_k)]


@api.get("/metrics", tags=["monitoring"])
def metrics(db: Session = Depends(get_db)):
    """Production monitors from the architecture: failures, latency, cost per call, correction rate per field."""
    by_status = dict(db.execute(select(Call.status, func.count()).group_by(Call.status)).all())
    processed = list(db.scalars(select(Call).where(Call.processed_at.is_not(None)).order_by(Call.id.desc()).limit(500)))
    audio = [c for c in processed if c.duration_s and c.processing_s and c.channel != Channel.WHATSAPP_TEXT]
    rtf = [c.processing_s / c.duration_s for c in audio]
    queue_lat = [(c.processed_at - c.created_at).total_seconds() - (c.processing_s or 0)
                 for c in processed if c.processed_at and c.created_at and c.created_at.tzinfo]
    costs = [c.usage.get("total_usd", 0) for c in processed if c.usage]
    fallbacks = sum(1 for c in processed if c.usage and c.usage.get("fallback_reason"))
    rows = db.execute(select(FieldCorrection.field, FieldCorrection.action, func.count())
                      .group_by(FieldCorrection.field, FieldCorrection.action)).all()
    per_field: dict[str, dict[str, int]] = {}
    for field, action, n in rows:
        per_field.setdefault(field, {})[action] = n
    correction_rate = {
        f: round((a.get("corrected", 0) + a.get("rejected", 0)) / max(1, sum(a.values())), 3)
        for f, a in per_field.items()
    }
    total = sum(by_status.values())
    return {
        "calls_by_status": by_status,
        "failure_rate": round(by_status.get(CallStatus.FAILED, 0) / total, 3) if total else 0.0,
        "processing_seconds_per_audio_minute": round(60 * sum(rtf) / len(rtf), 2) if rtf else None,
        "avg_queue_latency_s": round(sum(queue_lat) / len(queue_lat), 2) if queue_lat else None,
        "avg_cost_usd_per_call": round(sum(costs) / len(costs), 5) if costs else None,
        "llm_fallback_count": fallbacks,
        "field_review_actions": per_field,
        "correction_rate_per_field": correction_rate,
    }


# ---------------- webhooks ----------------


@hooks.post("/twilio/recording")
async def twilio_recording(request: Request, db: Session = Depends(get_db)):
    """Twilio RecordingStatusCallback. Only calls where the consent IVR prompt ran are recorded at all,
    so the presence of a recording is the consent record (store the IVR flag in production)."""
    form = await verify_twilio(request)
    if form.get("RecordingStatus", "completed") != "completed":
        return {"ignored": True}
    url = form.get("RecordingUrl")
    if not url:
        raise HTTPException(422, "RecordingUrl missing")
    sid = form.get("CallSid")
    if sid and db.scalar(select(Call).where(Call.external_id == sid)):
        return {"duplicate": True}
    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.get(url + ".wav")
        res.raise_for_status()
    path = _save_upload(res.content, ".wav")
    call = Call(lead_id=_lead_by_phone(db, form.get("To") or form.get("From")), channel=Channel.CALL,
                source_path=path, consent=True, external_id=sid)
    db.add(call)
    db.commit()
    enqueue(call.id)
    return {"call_id": call.id}


@hooks.get("/whatsapp", response_class=PlainTextResponse)
def whatsapp_verify(mode: str = Query(alias="hub.mode"), token: str = Query(alias="hub.verify_token"),
                    challenge: str = Query(alias="hub.challenge")):
    if mode == "subscribe" and token == get_settings().whatsapp_verify_token:
        return challenge
    raise HTTPException(403, "verification failed")


@hooks.post("/whatsapp")
async def whatsapp_inbound(payload: dict, db: Session = Depends(get_db)):
    """WhatsApp Business Cloud API webhook. Voice notes become audio calls; text messages are appended
    to the client's open text thread for the day (processed when the broker asks)."""
    created: list[int] = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for msg in value.get("messages", []):
                phone = msg.get("from")
                lead_id = _lead_by_phone(db, phone)
                if msg.get("type") == "audio":
                    path = await _download_wa_media(msg["audio"]["id"])
                    call = Call(lead_id=lead_id, channel=Channel.WHATSAPP_VOICE, source_path=path,
                                consent=True, external_id=msg.get("id"))
                    db.add(call)
                    db.commit()
                    enqueue(call.id)
                    created.append(call.id)
                elif msg.get("type") == "text":
                    line = f"client: {msg['text']['body']}"
                    thread = db.scalar(select(Call).where(
                        Call.channel == Channel.WHATSAPP_TEXT, Call.lead_id == lead_id,
                        Call.status == CallStatus.QUEUED, Call.external_id == f"wa-thread:{phone}"))
                    if thread:
                        thread.source_text = f"{thread.source_text}\n{line}"
                    else:
                        thread = Call(lead_id=lead_id, channel=Channel.WHATSAPP_TEXT, source_text=line,
                                      consent=True, external_id=f"wa-thread:{phone}")
                        db.add(thread)
                    db.commit()
                    created.append(thread.id)
    return {"calls": created}


async def _download_wa_media(media_id: str) -> str:
    s = get_settings()
    headers = {"Authorization": f"Bearer {s.whatsapp_access_token}"}
    async with httpx.AsyncClient(timeout=60, headers=headers) as client:
        meta = (await client.get(f"{s.whatsapp_graph_url}/{media_id}")).raise_for_status().json()
        media = (await client.get(meta["url"])).raise_for_status()
    return _save_upload(media.content, ".ogg")
