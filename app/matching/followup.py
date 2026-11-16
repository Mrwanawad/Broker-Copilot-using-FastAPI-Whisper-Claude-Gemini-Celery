"""Follow-up WhatsApp draft in Egyptian Arabic. Deterministic template: the broker edits and sends it;
nothing is sent automatically."""

from app.domain import CallSummary, Commitment, Speaker, UnitMatch
from app.text.numbers import format_egp

OBJECTION_LINES = {
    "price": "وراعيت إن الأسعار تكون في حدود الميزانية اللي اتكلمنا فيها",
    "down_payment": "واخترت وحدات بمقدم أقل عشان يكون مريح لحضرتك",
    "installment": "وفيه أنظمة سداد أطول عشان القسط يبقى أخف",
    "delivery": "ومنهم وحدات استلام قريب",
    "family_decision": "ولو تحب نعمل مكالمة مع المدام أو نرتب معاينة سوا يبقى أحسن",
    "timing": "ومفيش أي ضغط، خد وقتك وأنا متاح لأي سؤال",
}


def _unit_line(i: int, m: UnitMatch) -> str:
    plan = "كاش" if m.installment_years == 0 else f"مقدم {m.down_payment_pct:g}% وتقسيط {m.installment_years} سنين"
    delivery = "استلام فوري" if m.delivery == "ready" else f"استلام {m.delivery}"
    return f"{i}. {m.title} - {format_egp(m.price_egp)} - {plan} - {delivery}"


def draft_followup(summary: CallSummary, matches: list[UnitMatch], client_name: str | None = None) -> str:
    name = (client_name or "").split(" ")[0]
    lines = [f"أهلاً أستاذ {name}،" if name else "أهلاً بحضرتك،", "شكراً على وقتك في المكالمة."]
    if matches:
        lines.append("زي ما اتفقنا، دي أنسب وحدات لطلبك:")
        lines += [_unit_line(i, m) for i, m in enumerate(matches, 1)]
    extra = [OBJECTION_LINES[o.category] for o in summary.objections if o.category in OBJECTION_LINES]
    if extra:
        lines.append("، ".join(extra[:2]) + ".")
    visit: Commitment | None = next((c for c in summary.commitments if c.by == Speaker.BROKER and c.due), None)
    if visit:
        lines.append(f"وهكلم حضرتك يوم {visit.due.strftime('%d/%m')} زي ما اتفقنا.")
    else:
        lines.append("تحب نحدد ميعاد معاينة الأسبوع ده؟")
    return "\n".join(lines)
