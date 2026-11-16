"""Offline evaluation of the summarize -> CRM-fields stage against hand-labelled calls.

    python -m app.evaluation.run                 # summarizer from settings (auto)
    python -m app.evaluation.run --summarizer rules
    python -m app.evaluation.run --summarizer claude
    python -m app.evaluation.run --summarizer gemini
    python -m app.evaluation.run --summarizer gemini --replay data/eval/gemini_replay.jsonl  # offline, synthetic

Reports per-field accuracy, how many wrong values would have been auto-written (the number that
matters: a wrong auto_apply silently corrupts the CRM), and latency. Writes reports/eval_<name>.json.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from app.config import get_settings
from app.crm.mapping import propose_changes
from app.domain import FieldStatus
from app.evaluation.metrics import field_scores
from app.pipeline import parse_text

ROOT = Path(__file__).resolve().parents[2]


class _TextCall:  # the slice of a Call row that parse_text reads
    def __init__(self, text: str):
        self.source_text, self.broker = text, "Mona Nawy"


def predicted_fields(summary, changes) -> dict:
    out = {c.field: c.new for c in changes}
    out["areas"] = summary.needs.areas
    out["objections"] = [o.category for o in summary.objections]
    out["follow_up"] = out.pop("next_follow_up", None)
    out["stage"] = out.get("stage")
    return out


class _ReplayModels:
    """Stands in for the Gemini client: returns stored responses (one per case, in order)."""

    def __init__(self, path: Path):
        self.responses = [json.loads(line)["response"] for line in path.read_text(encoding="utf-8").splitlines()]

    def generate_content(self, **kwargs):
        meta = SimpleNamespace(prompt_token_count=0, candidates_token_count=0, thoughts_token_count=0)
        cand = SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))
        return SimpleNamespace(text=json.dumps(self.responses.pop(0), ensure_ascii=False), candidates=[cand],
                               usage_metadata=meta, response_id="replay")


def run(path: Path, summarizer: str | None, replay: Path | None = None) -> dict:
    if summarizer:
        os.environ["BC_SUMMARIZER"] = summarizer
        get_settings.cache_clear()
    if replay:
        from app.summarizer import gemini

        models = _ReplayModels(replay)
        gemini.get_client = lambda: SimpleNamespace(models=models)
        os.environ.setdefault("GOOGLE_API_KEY", "replay")
        get_settings.cache_clear()
    from app.summarizer.service import summarize

    threshold = get_settings().crm_confidence_threshold
    per_field: dict[str, list[bool]] = defaultdict(list)
    wrong_auto = total_auto = 0
    rows, latencies = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        transcript = parse_text(_TextCall(case["text"]))
        t0 = time.perf_counter()
        result = summarize(transcript, date.fromisoformat(case["call_date"]))
        latencies.append(time.perf_counter() - t0)
        summary = result["summary"]
        changes = propose_changes(summary, {}, threshold)
        pred = predicted_fields(summary, changes)
        scores = field_scores(case["gold"], pred)
        for f, ok in scores.items():
            per_field[f].append(ok)
        status = {c.field: c.status for c in changes}
        crm_name = {"areas": "preferred_areas"}
        for f, ok in scores.items():
            if status.get(crm_name.get(f, f)) == FieldStatus.AUTO_APPLY:
                total_auto += 1
                wrong_auto += not ok
        rows.append({"id": case["id"], "scores": scores, "summarizer": summary.summarizer,
                     "fallback": result["fallback_reason"], "headline": summary.headline,
                     "misses": {f: {"gold": case["gold"][f], "pred": pred.get(f)}
                                for f, ok in scores.items() if not ok}})
    all_scores = [ok for v in per_field.values() for ok in v]
    report = {
        "summarizer": rows[0]["summarizer"] if rows else None,
        "synthetic": bool(replay),
        "cases": len(rows),
        "field_accuracy": round(sum(all_scores) / len(all_scores), 4) if all_scores else None,
        "per_field_accuracy": {f: round(sum(v) / len(v), 3) for f, v in sorted(per_field.items())},
        "auto_applied_fields": total_auto,
        "wrong_auto_applied": wrong_auto,
        "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "rows": rows,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data/eval/calls.jsonl"))
    parser.add_argument("--summarizer", choices=["auto", "rules", "claude", "gemini"], default=None)
    parser.add_argument("--replay", default=None, help="stored Gemini responses (offline, synthetic)")
    args = parser.parse_args()
    report = run(Path(args.data), args.summarizer, Path(args.replay) if args.replay else None)
    name = (report["summarizer"] or "none").split(":")[0] + ("_synthetic" if args.replay else "")
    out = ROOT / "reports" / f"eval_{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summarizer: {report['summarizer']}  cases: {report['cases']}  field accuracy: {report['field_accuracy']}")
    print(f"auto-applied fields: {report['auto_applied_fields']}  wrong auto-applied: {report['wrong_auto_applied']}")
    for f, acc in report["per_field_accuracy"].items():
        print(f"  {f:22s} {acc:.2f}")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
