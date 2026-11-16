from datetime import datetime

from pydantic import BaseModel, Field

from app.domain import Needs


class CallOut(BaseModel):
    id: int
    lead_id: str | None
    channel: str
    status: str
    broker: str | None
    created_at: datetime
    processed_at: datetime | None = None
    duration_s: float | None = None
    processing_s: float | None = None
    error: str | None = None
    transcript: dict | None = None
    summary: dict | None = None
    crm_changes: list | None = None
    flagged_fields: list | None = None
    matches: list | None = None
    followup_draft: str | None = None
    timings: dict | None = None
    usage: dict | None = None
    approved_by: str | None = None
    approved_at: datetime | None = None

    model_config = {"from_attributes": True}


class CallBrief(BaseModel):
    id: int
    lead_id: str | None
    channel: str
    status: str
    created_at: datetime
    headline: str | None = None

    model_config = {"from_attributes": True}


class TextCallIn(BaseModel):
    lead_id: str | None = None
    text: str = Field(min_length=1, max_length=200_000)
    channel: str = "whatsapp_text"
    broker: str | None = None
    consent: bool = True  # chat threads the client sent to the broker


class ApproveIn(BaseModel):
    approved_fields: list[str] = Field(default_factory=list, description="needs_review fields the broker accepts")
    rejected_fields: list[str] = Field(default_factory=list, description="auto_apply fields the broker blocks")
    overrides: dict = Field(default_factory=dict, description="field -> corrected value")


class MatchIn(BaseModel):
    needs: Needs
    query: str = ""
    top_k: int = Field(3, ge=1, le=20)
