"""Prompt shared by the LLM summarizers (Claude primary, Gemini fallback)."""

from datetime import date

from app.domain import Transcript

SYSTEM_PROMPT = """You summarize real estate sales calls and WhatsApp threads between a broker and a client in Egypt. Conversations are mostly Egyptian Arabic, sometimes mixed with English.

Return the structured summary the broker's CRM needs. Rules:
- Use only what is said in the transcript. If something was not said, leave the field null with confidence 0. Do not guess a budget, area or date.
- Every filled field cites evidence: the segment index in brackets and a verbatim quote copied from that segment in its original language. Quotes must be exact substrings.
- Needs are the client's needs. A unit the broker offered counts only if the client accepted it.
- Money is an integer in EGP: "3 مليون ونص" is 3500000, "مليونين" is 2000000, "500 ألف" is 500000. A range "من 3 ل 4 مليون" fills budget_min_egp and budget_max_egp.
- areas: canonical English names (New Cairo, Sheikh Zayed, 6th of October, New Capital, North Coast, Ain Sokhna, Mostakbal City, Madinaty, El Shorouk, Maadi, Rehab).
- unit_type: apartment, villa, townhouse, twin house, duplex, penthouse, chalet or studio. payment_plan: cash, installments or mortgage. delivery: ready, within_1y, within_2y or any. purpose: living, investment, rental or summer.
- lead_stage: new (needs not known yet), qualified (needs known, no viewing agreed), viewing (a viewing or site visit is agreed or done), negotiation (price, discount or terms of a specific unit are being discussed), closed_won or closed_lost.
- Objections are concerns or hesitations the client raises. A plain preference (wanting installments, a certain area) is not an objection. Category is one of price, down_payment, installment, delivery, location, developer, trust, timing, family_decision, other. Suggest one concrete broker response for each.
- Commitments: who promised what, with a due date resolved against the call date when a day is mentioned: "النهارده" is the call date, "بكرة" the next day, a weekday name its next occurrence after the call date.
- confidence reflects how explicitly the client said it: stated clearly 0.9+, implied 0.5-0.7, unclear below 0.5.
- headline and next_action are short English sentences for the CRM activity log."""


def build_user_prompt(transcript: Transcript, call_date: date, lead_context: str = "") -> str:
    return (
        f"Call date: {call_date.isoformat()} ({call_date.strftime('%A')})\n"
        + (f"Existing CRM record:\n{lead_context}\n\n" if lead_context else "")
        + "Transcript (segment index, timestamp, speaker):\n"
        + transcript.as_text(with_index=True)
    )
