from datetime import date

from app.channels.text_threads import looks_like_whatsapp, parse_speaker_text, parse_whatsapp_export
from app.crm.mapping import apply_changes, propose_changes
from app.diarization.diarize import assign_roles, assign_turns
from app.domain import Evidence, FieldStatus, Segment, Speaker, Transcript
from app.evaluation.metrics import der, field_scores, wer_cer
from app.matching.followup import draft_followup
from app.matching.matcher import Matcher, load_inventory
from app.summarizer.rules import extract
from app.summarizer.service import verify_evidence
from tests.conftest import CALL_TEXT

WA = """12/11/2026, 18:02 - Mona Nawy: مساء الخير، ده رقمي من ناوي
12/11/2026, 18:10 - Yasser: عايز تاون هاوس في العاصمة
تكملة الرسالة
12/11/2026, 18:11 - Yasser: <Media omitted>"""


def test_whatsapp_export():
    assert looks_like_whatsapp(WA)
    t = parse_whatsapp_export(WA, ["Mona Nawy"])
    assert [s.speaker for s in t.segments] == [Speaker.BROKER, Speaker.CLIENT]
    assert "تكملة" in t.segments[1].text
    assert t.segments[1].start == 480


def test_rules_extract():
    t = parse_speaker_text(CALL_TEXT)
    s = extract(t, date(2026, 11, 12))
    assert s.needs.areas == ["New Cairo"]
    assert s.needs.unit_type.value == "apartment"
    assert s.needs.bedrooms.value == 3
    assert s.needs.budget_max_egp.value == 4_500_000
    assert s.needs.installment_years.value == 8
    assert s.needs.payment_plan.value == "installments"
    assert [o.category for o in s.objections] == ["price"]
    assert s.follow_up_date.value == "2026-11-19"
    assert s.lead_stage.value == "viewing"
    assert all(e.segment is not None for e in s.needs.budget_max_egp.evidence)


def test_budget_and_down_payment_percent_in_one_message():
    t = parse_speaker_text("client: الميزانية حوالي 4 مليون ونص، والمقدم مش أكتر من 10%")
    s = extract(t, date(2026, 11, 17))
    assert s.needs.budget_max_egp.value == 4_500_000
    assert s.needs.down_payment_egp.value is None

    t = parse_speaker_text("client: ميزانيتي 6 مليون والمقدم 600 ألف")
    s = extract(t, date(2026, 11, 17))
    assert s.needs.budget_max_egp.value == 6_000_000
    assert s.needs.down_payment_egp.value == 600_000


def test_verify_evidence_penalizes_invented_quotes():
    t = parse_speaker_text(CALL_TEXT)
    s = extract(t, date(2026, 11, 12))
    s.needs.budget_max_egp.value = 9_000_000
    s.needs.budget_max_egp.confidence = 0.95
    s.needs.budget_max_egp.evidence = [Evidence(segment=2, quote="ميزانيتي 9 مليون")]
    flagged = verify_evidence(s, t)
    assert "needs.budget_max_egp" in flagged
    assert s.needs.budget_max_egp.confidence < 0.75


def test_crm_policy():
    s = extract(parse_speaker_text(CALL_TEXT), date(2026, 11, 12))
    lead = {"budget_max": 6_000_000, "unit_type": "apartment", "preferred_areas": []}
    changes = {c.field: c for c in propose_changes(s, lead, 0.75)}
    assert changes["budget_max"].status == FieldStatus.NEEDS_REVIEW  # overwrites a stored value
    assert changes["unit_type"].status == FieldStatus.UNCHANGED
    assert changes["installment_years"].status == FieldStatus.AUTO_APPLY
    updated = apply_changes(lead, list(changes.values()))
    assert updated["budget_max"] == 6_000_000 and updated["installment_years"] == 8
    updated = apply_changes(lead, list(changes.values()), {"budget_max"}, {"bedrooms": "4"})
    assert updated["budget_max"] == 4_500_000 and updated["bedrooms"] == 4


def test_matcher_respects_constraints():
    s = extract(parse_speaker_text(CALL_TEXT), date(2026, 11, 12))
    matches = Matcher(load_inventory("data/inventory.json")).match(s.needs, top_k=3)
    assert matches
    for m in matches:
        assert m.price_egp <= 4_500_000 * 1.15
        assert m.unit_type == "apartment" and abs(m.bedrooms - 3) <= 1  # never a studio for a 3BR family
    draft = draft_followup(s, matches, "Ahmed Samir")
    assert draft.startswith("أهلاً أستاذ Ahmed") and matches[0].title in draft


def test_diarization_role_assignment():
    t = Transcript(segments=[
        Segment(start=0, end=3, text="مساء الخير معاك منى من ناوي"),
        Segment(start=3, end=6, text="أهلاً انا بدور على شقة ميزانيتي 3 مليون"),
        Segment(start=6, end=9, text="عندنا وحدات هبعتلك اللوكيشن"),
    ])
    turns = [(0, 3, "SPEAKER_01"), (3, 6, "SPEAKER_00"), (6, 9, "SPEAKER_01")]
    t = assign_roles(assign_turns(t, turns))
    assert [s.speaker for s in t.segments] == [Speaker.BROKER, Speaker.CLIENT, Speaker.BROKER]


def test_metrics():
    assert wer_cer("أنا عايز شقة", "انا عايز شقه")["wer"] == 0.0
    ref = [(0, 5, "A"), (5, 10, "B")]
    assert der(ref, [(0, 5, "x"), (5, 10, "y")]) == 0.0
    assert der(ref, [(0, 10, "x")]) > 0.3
    assert field_scores({"budget_max": 4_500_000, "areas": ["A", "B"]},
                        {"budget_max": 4_550_000, "areas": ["B", "A"]}) == {"budget_max": True, "areas": True}
