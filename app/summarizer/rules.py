"""Offline, deterministic extractor: transcript -> CallSummary.

Runs when Claude is not configured, and is the baseline the LLM summarizer is evaluated against.
Every value carries the segment and quote it came from, same contract as the LLM path.
"""

import re
from datetime import date, timedelta

from app.domain import (
    CallSummary,
    Commitment,
    Evidence,
    Field_,
    Needs,
    Objection,
    Segment,
    Sentiment,
    Speaker,
    Transcript,
)
from app.summarizer.playbook import (
    BROKER_COMMITMENT_CUES,
    CLIENT_COMMITMENT_CUES,
    NEGATIVE_CUES,
    OBJECTION_CUES,
    PAYMENT_CUES,
    POSITIVE_CUES,
    SUGGESTED_RESPONSES,
)
from app.text import dates
from app.text.gazetteer import find_areas, find_delivery, find_purposes, find_unit_types
from app.text.normalize import normalize_search
from app.text.numbers import find_amounts, find_bedrooms, find_percents, find_years

BUDGET_CUES = ["ميزاني", "budget", "معايا", "اقدر ادفع", "في حدود", "حدود", "لحد", "مش اكتر من", "max", "اقصى"]
DOWN_CUES = ["مقدم", "down payment", "داون"]
# clause breaks; commas/dots inside numbers ("4,250,000", "7.5M") don't split
_CLAUSE = re.compile(r"[،؛;\n]|,(?!\d{3})|\.(?!\d)")
RANGE_CUES = [" ل", " الى ", " لحد ", " to ", "-"]


def _has(text: str, cues: list[str]) -> bool:
    norm = normalize_search(text)
    return any(normalize_search(c) in norm for c in cues)


def _ev(i: int, seg: Segment) -> list[Evidence]:
    return [Evidence(segment=i, quote=seg.text[:200])]


def _set(field: Field_, value, conf: float, i: int, seg: Segment) -> None:
    """First mention wins unless a later one is more confident (clients revise budgets mid-call)."""
    if field.value is None or conf > field.confidence:
        field.value, field.confidence, field.evidence = value, conf, _ev(i, seg)


def extract(transcript: Transcript, call_date: date | None = None) -> CallSummary:
    call_date = call_date or date.today()
    segs = transcript.segments
    needs = Needs()
    objections: list[Objection] = []
    commitments: list[Commitment] = []
    pos = neg = 0
    stage_hints: set[str] = set()
    broker_areas: list[tuple[str, int, Segment]] = []

    for i, seg in enumerate(segs):
        text = seg.text
        is_client = seg.speaker != Speaker.BROKER
        # areas and types: client statements are strong, broker offers the client accepted are weaker
        for area in find_areas(text):
            if area not in needs.areas and (is_client or _has(text, ["عايز", "تمام", "مناسب"])):
                needs.areas.append(area)
                needs.area_evidence += _ev(i, seg)
            elif not is_client:
                broker_areas.append((area, i, seg))
        types = find_unit_types(text)
        if types:
            _set(needs.unit_type, types[0], 0.85 if is_client else 0.6, i, seg)
        beds = find_bedrooms(text)
        if beds:
            _set(needs.bedrooms, beds[0], 0.85 if is_client else 0.6, i, seg)
        purposes = find_purposes(text)
        if purposes and is_client:
            _set(needs.purpose, purposes[0], 0.7, i, seg)
        delivery = find_delivery(text)
        if delivery:
            _set(needs.delivery, delivery[0], 0.75 if is_client else 0.55, i, seg)

        # per clause, so "budget 4.5M, down payment max 10%" doesn't make 4.5M the down payment
        budget_amounts = []
        carry = False  # "المقدم مش كبير، أقصى حاجة 400 ألف": the cue's amount sits in the next clause
        for clause in _CLAUSE.split(text):
            amounts = find_amounts(clause)
            low = clause.lower()
            # amounts after cue_at are the down payment, amounts before it the budget; None = no cue
            cue_at = min((low.find(c) for c in DOWN_CUES if c in low), default=None)
            if cue_at is None and carry and not _has(clause, ["ميزاني", "budget"]):
                cue_at = -1
            has_pct = cue_at is not None and bool(find_percents(clause[max(cue_at, 0):]))
            carry = cue_at is not None and not amounts and not has_pct
            down = [a.value for a in amounts if cue_at is not None and a.start > cue_at]
            if down:
                _set(needs.down_payment_egp, down[0], 0.85 if is_client else 0.55, i, seg)
            elif has_pct and is_client:
                needs.down_payment_egp.evidence = _ev(i, seg)  # percentage noted; amount resolved by matcher
            if is_client:
                budget_amounts += [a.value for a in amounts if cue_at is None or a.start < cue_at]
        if budget_amounts:
            conf = 0.9 if _has(text, BUDGET_CUES) else 0.65
            if len(budget_amounts) >= 2:
                _set(needs.budget_min_egp, min(budget_amounts), conf, i, seg)
                _set(needs.budget_max_egp, max(budget_amounts), conf, i, seg)
            else:
                _set(needs.budget_max_egp, budget_amounts[0], conf, i, seg)

        for plan, cues in PAYMENT_CUES.items():
            if _has(text, cues):
                _set(needs.payment_plan, plan, 0.85 if is_client else 0.6, i, seg)
                break
        years = find_years(text)
        if years and (_has(text, PAYMENT_CUES["installments"]) or _has(text, ["على"])):
            _set(needs.installment_years, years[0], 0.8 if is_client else 0.6, i, seg)

        if is_client:
            for category, cues in OBJECTION_CUES.items():
                if _has(text, cues) and not any(o.category == category for o in objections):
                    objections.append(Objection(
                        category=category, text=text[:200], evidence=_ev(i, seg),
                        suggested_response=SUGGESTED_RESPONSES[category],
                    ))
            pos += _has(text, POSITIVE_CUES)
            neg += _has(text, NEGATIVE_CUES)

        cues = BROKER_COMMITMENT_CUES if seg.speaker == Speaker.BROKER else CLIENT_COMMITMENT_CUES
        if _has(text, cues):
            commitments.append(Commitment(
                by=seg.speaker if seg.speaker != Speaker.UNKNOWN else Speaker.CLIENT,
                text=text[:200], due=dates.resolve(text, call_date), evidence=_ev(i, seg),
            ))
        if _has(text, ["معاينه", "معاينة", "site visit", "زياره"]):
            stage_hints.add("viewing")
        if _has(text, ["خصم", "discount", "اخر سعر", "تفاوض"]):
            stage_hints.add("negotiation")
        if _has(text, ["نحجز", "احجز", "حجز", "reservation", "عربون"]):
            stage_hints.add("reservation")

    # the broker restating the client's area ("الشاليه في الساحل") counts when the client named none
    if not needs.areas:
        for area, i, seg in broker_areas:
            if area not in needs.areas:
                needs.areas.append(area)
                needs.area_evidence += _ev(i, seg)

    # stage
    if "reservation" in stage_hints:
        stage, stage_conf = "negotiation", 0.7
    elif "negotiation" in stage_hints:
        stage, stage_conf = "negotiation", 0.65
    elif "viewing" in stage_hints:
        stage, stage_conf = "viewing", 0.65
    elif needs.budget_max_egp.value and needs.areas:
        stage, stage_conf = "qualified", 0.7
    else:
        stage, stage_conf = "new", 0.5

    # follow-up: earliest dated commitment, else 2 days out
    dated = sorted((c for c in commitments if c.due and c.due > call_date), key=lambda c: c.due)
    if dated:
        follow = Field_(value=dated[0].due.isoformat(), confidence=0.8, evidence=dated[0].evidence)
    else:
        follow = Field_(value=(call_date + timedelta(days=2)).isoformat(), confidence=0.4)

    sentiment = Sentiment.NEUTRAL
    if pos > neg:
        sentiment = Sentiment.POSITIVE
    elif neg > pos:
        sentiment = Sentiment.NEGATIVE

    next_action = _next_action(needs, objections, commitments, stage)
    headline = _headline(needs, objections, stage)
    return CallSummary(
        headline=headline, needs=needs, objections=objections, commitments=commitments,
        sentiment=sentiment, sentiment_confidence=0.6 if pos != neg else 0.4,
        lead_stage=Field_(value=stage, confidence=stage_conf), next_action=next_action,
        follow_up_date=follow, summarizer="rules",
    )


def _next_action(needs: Needs, objections: list[Objection], commitments: list[Commitment], stage: str) -> str:
    broker_todo = [c for c in commitments if c.by == Speaker.BROKER]
    if broker_todo:
        return f"Do what was promised: {broker_todo[0].text[:120]}"
    if objections:
        return SUGGESTED_RESPONSES[objections[0].category]
    if stage == "qualified":
        return "Send 3 matched units on WhatsApp and propose a site visit."
    if not needs.budget_max_egp.value:
        return "Qualify budget and payment method on the next call."
    return "Follow up with matched units."


def _headline(needs: Needs, objections: list[Objection], stage: str) -> str:
    from app.text.numbers import format_egp

    parts = []
    if needs.unit_type.value:
        beds = f"{needs.bedrooms.value}BR " if needs.bedrooms.value else ""
        parts.append(f"{beds}{needs.unit_type.value}")
    if needs.areas:
        parts.append("in " + "/".join(needs.areas))
    if needs.budget_max_egp.value:
        parts.append(f"up to {format_egp(needs.budget_max_egp.value)}")
    if needs.payment_plan.value:
        plan = needs.payment_plan.value
        if needs.installment_years.value:
            plan += f" over {needs.installment_years.value}y"
        parts.append(f"({plan})")
    head = "Client wants " + " ".join(parts) if parts else "Call with client"
    if objections:
        head += f"; concerns: {', '.join(o.category for o in objections[:3])}"
    return f"{head}. Stage: {stage}."
