from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class CallStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    REVIEW = "review"  # processed, waiting for the broker to approve CRM changes
    APPROVED = "approved"
    FAILED = "failed"


class Lead(Base):
    """CRM mirror. `data` holds the CRM fields so new fields need no migration."""

    __tablename__ = "leads"
    lead_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(200))
    phone: Mapped[str | None] = mapped_column(String(32), index=True)
    owner: Mapped[str | None] = mapped_column(String(100))
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    def as_dict(self) -> dict:
        return {"lead_id": self.lead_id, "name": self.name, "phone": self.phone, "owner": self.owner,
                **(self.data or {})}


class Call(Base):
    __tablename__ = "calls"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lead_id: Mapped[str | None] = mapped_column(ForeignKey("leads.lead_id"), index=True)
    channel: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default=CallStatus.QUEUED, index=True)
    source_path: Mapped[str | None] = mapped_column(String(500))
    source_text: Mapped[str | None] = mapped_column(Text)
    external_id: Mapped[str | None] = mapped_column(String(128), index=True)  # Twilio CallSid / WA message id
    consent: Mapped[bool] = mapped_column(default=False)
    broker: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_s: Mapped[float | None] = mapped_column(Float)
    processing_s: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)

    transcript: Mapped[dict | None] = mapped_column(JSON)
    summary: Mapped[dict | None] = mapped_column(JSON)
    crm_changes: Mapped[list | None] = mapped_column(JSON)
    flagged_fields: Mapped[list | None] = mapped_column(JSON)
    matches: Mapped[list | None] = mapped_column(JSON)
    followup_draft: Mapped[str | None] = mapped_column(Text)
    timings: Mapped[dict | None] = mapped_column(JSON)  # per-stage seconds
    usage: Mapped[dict | None] = mapped_column(JSON)  # LLM tokens, ASR seconds, cost estimate

    approved_by: Mapped[str | None] = mapped_column(String(100))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class FieldCorrection(Base):
    """One row per CRM field the broker reviewed. Feeds the 'broker correction rate per field' monitor."""

    __tablename__ = "field_corrections"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[int] = mapped_column(ForeignKey("calls.id"), index=True)
    field: Mapped[str] = mapped_column(String(64), index=True)
    proposed: Mapped[str | None] = mapped_column(Text)
    final: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    action: Mapped[str] = mapped_column(String(16))  # accepted | corrected | rejected | auto
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
