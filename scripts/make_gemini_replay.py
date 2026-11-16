"""Build SYNTHETIC Gemini responses for the eval set, so the Gemini path can be evaluated offline.

The live Gemini run hit the free-tier quota after 3 of 5 cases. Each synthetic response starts from
the rules extraction of the case and applies the deviations seen in the live gemini-3.5-flash runs
(2026-11-16): conservative lead stage, client's "also" area kept, follow-up dates resolved as the next
weekday occurrence. Cases with no live output (c4) keep the rules output unchanged.

These are not real model outputs. The eval report marks them with "synthetic": true.

    python scripts/make_gemini_replay.py   # writes data/eval/gemini_replay.jsonl
"""

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.domain import Field_  # noqa: E402
from app.evaluation.run import _TextCall  # noqa: E402
from app.pipeline import parse_text  # noqa: E402
from app.summarizer import rules  # noqa: E402

# deviations observed in the live gemini-3.5-flash runs, per case
OBSERVED = {
    "c1": {"stage": "qualified", "follow_up": None},
    "c2": {"areas": ["6th of October", "Sheikh Zayed"]},
    "c3": {"stage": "qualified", "objections": ["delivery"], "follow_up": "2026-11-21"},
    "c5": {"stage": "qualified"},
}


def synthesize(case: dict) -> dict:
    transcript = parse_text(_TextCall(case["text"]))
    s = rules.extract(transcript, date.fromisoformat(case["call_date"]))
    obs = OBSERVED.get(case["id"], {})
    if "stage" in obs:
        s.lead_stage = Field_(value=obs["stage"], confidence=0.85, evidence=s.lead_stage.evidence)
    if "areas" in obs:
        s.needs.areas = obs["areas"]
    if "objections" in obs:
        s.objections = [o for o in s.objections if o.category in obs["objections"]]
    if "follow_up" in obs:
        fu = obs["follow_up"]
        s.follow_up_date = Field_(value=fu, confidence=0.8 if fu else 0.0,
                                  evidence=s.follow_up_date.evidence if fu else [])
    s.summarizer = "gemini:gemini-3.5-flash"
    return {"id": case["id"], "synthetic": True, "response": s.model_dump(mode="json")}


def main() -> None:
    cases = [json.loads(line) for line in (ROOT / "data/eval/calls.jsonl").read_text(encoding="utf-8").splitlines()]
    out = ROOT / "data/eval/gemini_replay.jsonl"
    out.write_text("\n".join(json.dumps(synthesize(c), ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} synthetic responses -> {out}")


if __name__ == "__main__":
    main()
