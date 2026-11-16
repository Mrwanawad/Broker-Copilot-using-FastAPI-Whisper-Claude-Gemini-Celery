"""End-to-end audio evaluation: preprocess -> ASR -> diarization -> summary, against the reference
files written by make_demo_audio.py.

Reports WER/CER, DER, speaker-role accuracy, CRM field accuracy from audio, and processing seconds
per audio minute. Writes reports/eval_audio_<model>.json.

    python scripts/eval_audio.py --whisper-model base
    python scripts/eval_audio.py --whisper-model small --diarizer pyannote --synthetic-pyannote

--synthetic-pyannote: no access to the gated pyannote models, so its turns are SYNTHESIZED from the
reference turns (seeded +-0.25 s boundary jitter, anonymous SPEAKER_0x labels in random order). ASR,
role assignment and summary still run for real. The report is marked "synthetic": true.
"""

import argparse
import json
import os
import random
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--summarizer", default="rules")
    parser.add_argument("--diarizer", default="heuristic")
    parser.add_argument("--synthetic-pyannote", action="store_true", help="synthesized pyannote turns")
    args = parser.parse_args()
    os.environ.update(BC_WHISPER_MODEL=args.whisper_model, BC_SUMMARIZER=args.summarizer,
                      BC_DIARIZER=args.diarizer, BC_ASR_BACKEND="faster_whisper")

    from app.crm.mapping import propose_changes
    from app.evaluation.metrics import der, field_scores, wer_cer
    from app.evaluation.run import predicted_fields
    from app.pipeline import transcribe_audio
    from app.summarizer.service import summarize

    if args.synthetic_pyannote:
        from app.diarization import diarize

        refs = {}

        def synthetic_turns(wav: Path, num_speakers: int | None = 2) -> list:
            ref = next(iter(refs.values()))  # the case being evaluated (wav here is the preprocessed copy)
            rng = random.Random(ref["id"])
            labels = dict(zip(["broker", "client"], rng.sample(["SPEAKER_00", "SPEAKER_01"], 2), strict=True))
            turns = []
            for t in ref["turns"]:
                start = max(0.0, t["start"] + rng.uniform(-0.25, 0.25))
                turns.append((start, max(start + 0.2, t["end"] + rng.uniform(-0.25, 0.25)), labels[t["speaker"]]))
            return turns

        diarize.pyannote_turns = synthetic_turns

    rows = []
    for ref_path in sorted((ROOT / "data/demo_audio").glob("*.ref.json")):
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        wav = ref_path.with_name(ref["id"] + ".wav")
        if args.synthetic_pyannote:
            refs.clear()
            refs[ref["id"]] = ref
        timings: dict = {}
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            transcript = transcribe_audio(wav, Path(tmp), timings)
        elapsed = time.perf_counter() - t0
        ref_text = " ".join(t["text"] for t in ref["turns"])
        hyp_text = " ".join(s.text for s in transcript.segments)
        asr = wer_cer(ref_text, hyp_text)
        ref_turns = [(t["start"], t["end"], t["speaker"]) for t in ref["turns"]]
        hyp_turns = [(s.start, s.end, s.speaker.value) for s in transcript.segments]
        # role accuracy: speaker label of each hypothesis segment vs the reference turn it overlaps most
        correct = 0
        for s in transcript.segments:
            best = max(ref["turns"], key=lambda t, s=s: min(s.end, t["end"]) - max(s.start, t["start"]))
            correct += best["speaker"] == s.speaker.value
        result = summarize(transcript, date.fromisoformat(ref["call_date"]))
        changes = propose_changes(result["summary"], {}, 0.75)
        scores = field_scores(ref["gold"], predicted_fields(result["summary"], changes))
        rows.append({
            "id": ref["id"], **asr, "der": der(ref_turns, hyp_turns),
            "role_accuracy": round(correct / max(1, len(transcript.segments)), 3),
            "field_accuracy": round(sum(scores.values()) / len(scores), 3), "field_scores": scores,
            "audio_s": round(transcript.duration_s, 1), "processing_s": round(elapsed, 1),
            "sec_per_audio_min": round(60 * elapsed / max(1.0, transcript.duration_s), 1), "timings": timings,
            "hypothesis": transcript.as_text(),
        })
        print(f"{ref['id']}: WER {asr['wer']:.2f} CER {asr['cer']:.2f} DER {rows[-1]['der']:.2f} "
              f"roles {rows[-1]['role_accuracy']:.2f} fields {rows[-1]['field_accuracy']:.2f} "
              f"({rows[-1]['sec_per_audio_min']}s per audio min)")

    def mean(key: str) -> float:
        return round(sum(r[key] for r in rows) / len(rows), 3) if rows else 0.0

    report = {"whisper_model": args.whisper_model, "diarizer": args.diarizer, "summarizer": args.summarizer,
              "synthetic": args.synthetic_pyannote,
              **{k: mean(k) for k in ("wer", "cer", "der", "role_accuracy", "field_accuracy", "sec_per_audio_min")},
              "rows": rows}
    suffix = "_pyannote_synthetic" if args.synthetic_pyannote else ""
    out = ROOT / "reports" / f"eval_audio_{args.whisper_model}{suffix}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print({k: v for k, v in report.items() if k != "rows"})


if __name__ == "__main__":
    main()
