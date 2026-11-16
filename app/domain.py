"""Shared data contracts between pipeline stages, API and UI."""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class Channel(StrEnum):
    CALL = "call"
    WHATSAPP_VOICE = "whatsapp_voice"
    WHATSAPP_TEXT = "whatsapp_text"
    UPLOAD = "upload"


class Speaker(StrEnum):
    BROKER = "broker"
    CLIENT = "client"
    UNKNOWN = "unknown"


class Word(BaseModel):
    start: float
    end: float
    text: str


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Speaker = Speaker.UNKNOWN
    speaker_raw: str | None = None  # diarizer label before role assignment (SPEAKER_00)
    confidence: float | None = None
    words: list[Word] = Field(default_factory=list)


class Transcript(BaseModel):
    language: str = "ar"
    duration_s: float = 0.0
    segments: list[Segment] = Field(default_factory=list)
    asr_model: str | None = None

    def as_text(self, with_speakers: bool = True, with_index: bool = False) -> str:
        lines = []
        for i, s in enumerate(self.segments):
            prefix = f"[{i}] " if with_index else ""
            who = f"{s.speaker.value}: " if with_speakers else ""
            lines.append(f"{prefix}[{fmt_ts(s.start)}] {who}{s.text}")
        return "\n".join(lines)


def fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ---------- structured summary ----------


class Evidence(BaseModel):
    segment: int | None = Field(None, description="Index of the transcript segment the quote comes from")
    quote: str = Field(description="Verbatim quote from the transcript, in its original language")


class Field_(BaseModel):
    """A value plus the confidence and evidence behind it."""

    value: str | int | float | None = None
    confidence: float = Field(0.0, ge=0, le=1)
    evidence: list[Evidence] = Field(default_factory=list)


class PaymentPlan(StrEnum):
    CASH = "cash"
    INSTALLMENTS = "installments"
    MORTGAGE = "mortgage"
    UNKNOWN = "unknown"


class Needs(BaseModel):
    areas: list[str] = Field(default_factory=list, description="Preferred areas, normalized English names")
    unit_type: Field_ = Field(default_factory=Field_)  # apartment, villa, townhouse, twin house, duplex, chalet, studio
    bedrooms: Field_ = Field(default_factory=Field_)
    budget_min_egp: Field_ = Field(default_factory=Field_)
    budget_max_egp: Field_ = Field(default_factory=Field_)
    down_payment_egp: Field_ = Field(default_factory=Field_)
    payment_plan: Field_ = Field(default_factory=Field_)  # PaymentPlan value
    installment_years: Field_ = Field(default_factory=Field_)
    delivery: Field_ = Field(default_factory=Field_)  # ready | within_1y | within_2y | any
    purpose: Field_ = Field(default_factory=Field_)  # living | investment | rental | summer
    area_evidence: list[Evidence] = Field(default_factory=list)


class Objection(BaseModel):
    category: str = Field(description=(
        "price | down_payment | installment | delivery | location | developer | trust | timing"
        " | family_decision | other"
    ))
    text: str
    evidence: list[Evidence] = Field(default_factory=list)
    suggested_response: str | None = None


class Commitment(BaseModel):
    by: Speaker
    text: str
    due: date | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class CallSummary(BaseModel):
    headline: str = Field(description="One-line summary for the CRM activity log")
    needs: Needs = Field(default_factory=Needs)
    objections: list[Objection] = Field(default_factory=list)
    commitments: list[Commitment] = Field(default_factory=list)
    sentiment: Sentiment = Sentiment.NEUTRAL
    sentiment_confidence: float = 0.5
    # new | qualified | viewing | negotiation | closed_won | closed_lost
    lead_stage: Field_ = Field(default_factory=Field_)
    next_action: str = ""
    follow_up_date: Field_ = Field(default_factory=Field_)  # ISO date
    summarizer: str = "rules"


# ---------- CRM ----------


class FieldStatus(StrEnum):
    AUTO_APPLY = "auto_apply"  # high confidence, unchanged elsewhere
    NEEDS_REVIEW = "needs_review"  # low confidence or overwrites an existing value
    UNCHANGED = "unchanged"


class CRMChange(BaseModel):
    field: str
    old: str | int | float | list | None = None
    new: str | int | float | list | None = None
    confidence: float
    status: FieldStatus
    evidence: list[Evidence] = Field(default_factory=list)


class UnitMatch(BaseModel):
    unit_id: str
    title: str
    compound: str
    area: str
    unit_type: str
    bedrooms: int
    price_egp: int
    down_payment_pct: float
    installment_years: int
    delivery: str
    score: float
    reasons: list[str] = Field(default_factory=list)
