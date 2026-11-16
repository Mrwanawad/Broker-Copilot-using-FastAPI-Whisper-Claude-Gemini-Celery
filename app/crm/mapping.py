"""CallSummary -> proposed CRM field changes, each with confidence, evidence and an apply status.

Policy (architecture: "auto-writing low-confidence CRM fields: broker confirms"):
- confidence >= threshold and the CRM field is empty      -> auto_apply
- confidence >= threshold but it overwrites a stored value -> needs_review (clients revise budgets;
  the broker decides which number is current)
- confidence < threshold                                   -> needs_review
- same value as stored                                     -> unchanged
"""

from app.domain import CallSummary, CRMChange, Field_, FieldStatus

# CRM field -> path into the summary
FIELD_MAP: dict[str, str] = {
    "unit_type": "needs.unit_type",
    "bedrooms": "needs.bedrooms",
    "budget_min": "needs.budget_min_egp",
    "budget_max": "needs.budget_max_egp",
    "down_payment": "needs.down_payment_egp",
    "payment_method": "needs.payment_plan",
    "installment_years": "needs.installment_years",
    "delivery_preference": "needs.delivery",
    "purpose": "needs.purpose",
    "stage": "lead_stage",
    "next_follow_up": "follow_up_date",
}
NUMERIC = {"bedrooms", "budget_min", "budget_max", "down_payment", "installment_years"}


def _get(summary: CallSummary, path: str) -> Field_:
    obj = summary
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _coerce(field: str, value):
    if value is None or value == "":
        return None
    if field in NUMERIC:
        try:
            return int(float(str(value).replace(",", "")))
        except ValueError:
            return None
    return str(value)


def _status(old, new, confidence: float, threshold: float) -> FieldStatus:
    if old == new:
        return FieldStatus.UNCHANGED
    if confidence >= threshold and old in (None, "", []):
        return FieldStatus.AUTO_APPLY
    return FieldStatus.NEEDS_REVIEW


def propose_changes(summary: CallSummary, lead: dict, threshold: float) -> list[CRMChange]:
    changes: list[CRMChange] = []
    for crm_field, path in FIELD_MAP.items():
        f = _get(summary, path)
        new = _coerce(crm_field, f.value)
        if new is None:
            continue
        old = lead.get(crm_field)
        changes.append(CRMChange(
            field=crm_field, old=old, new=new, confidence=round(f.confidence, 3),
            status=_status(old, new, f.confidence, threshold), evidence=f.evidence,
        ))
    if summary.needs.areas:
        old_areas = lead.get("preferred_areas") or []
        merged = list(dict.fromkeys([*old_areas, *summary.needs.areas]))
        conf = 0.85 if summary.needs.area_evidence else 0.5
        status = FieldStatus.UNCHANGED if merged == old_areas else (
            FieldStatus.AUTO_APPLY if conf >= threshold else FieldStatus.NEEDS_REVIEW)
        changes.append(CRMChange(field="preferred_areas", old=old_areas, new=merged, confidence=conf,
                                 status=status, evidence=summary.needs.area_evidence))
    # activity-log fields are append-only notes, always safe to write
    changes.append(CRMChange(field="last_call_summary", old=lead.get("last_call_summary"), new=summary.headline,
                             confidence=1.0, status=FieldStatus.AUTO_APPLY))
    if summary.objections:
        changes.append(CRMChange(
            field="open_objections", old=lead.get("open_objections"),
            new=[o.category for o in summary.objections], confidence=0.9, status=FieldStatus.AUTO_APPLY,
            evidence=[e for o in summary.objections for e in o.evidence][:5],
        ))
    return changes


def apply_changes(lead: dict, changes: list[CRMChange], approved_fields: set[str] | None = None,
                  overrides: dict | None = None) -> dict:
    """New lead record. auto_apply fields are written; needs_review only when the broker approved them.
    `overrides` are broker corrections (field -> value) and always win."""
    updated = dict(lead)
    approved_fields = approved_fields or set()
    for c in changes:
        if c.status == FieldStatus.AUTO_APPLY or (c.status == FieldStatus.NEEDS_REVIEW and c.field in approved_fields):
            updated[c.field] = c.new
    for field, value in (overrides or {}).items():
        updated[field] = _coerce(field, value) if field in NUMERIC else value
    return updated
