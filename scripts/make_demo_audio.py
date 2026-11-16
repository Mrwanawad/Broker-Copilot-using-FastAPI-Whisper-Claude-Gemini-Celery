"""Synthesize two-speaker Egyptian-Arabic demo calls from the eval transcripts (edge-tts, ar-EG voices).

Writes data/demo_audio/<id>.wav plus <id>.ref.json (reference text and speaker turns), which
scripts/eval_audio.py uses for WER/CER and DER. Synthetic speech is cleaner than real phone audio,
so these numbers are an upper bound; real calls need the audited-sample monitor.

    uv pip install edge-tts
    python scripts/make_demo_audio.py [--ids c1 c3]
"""

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.channels.text_threads import looks_like_whatsapp, parse_speaker_text  # noqa: E402
from app.domain import Speaker  # noqa: E402

VOICES = {Speaker.BROKER: "ar-EG-SalmaNeural", Speaker.CLIENT: "ar-EG-ShakirNeural"}
GAP_S = 0.5


async def tts(text: str, voice: str, out: Path) -> None:
    import edge_tts

    for attempt in range(4):  # the free endpoint drops requests now and then
        try:
            await edge_tts.Communicate(text, voice).save(str(out))
            return
        except edge_tts.exceptions.NoAudioReceived:
            await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"TTS failed for: {text}")


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "default=nw=1:nk=1", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


async def build(case: dict, out_dir: Path) -> None:
    transcript = parse_speaker_text(case["text"])
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        parts, turns, t = [], [], 0.0
        for i, seg in enumerate(transcript.segments):
            mp3 = tmp_dir / f"{i}.mp3"
            await tts(seg.text, VOICES[seg.speaker], mp3)
            wav = tmp_dir / f"{i}.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-af",
                            f"apad=pad_dur={GAP_S}", "-ar", "16000", "-ac", "1", str(wav)], check=True)
            d = duration(wav)
            turns.append({"start": round(t, 3), "end": round(t + d - GAP_S, 3), "speaker": seg.speaker.value,
                          "text": seg.text})
            parts.append(wav)
            t += d
        listing = tmp_dir / "list.txt"
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        out = out_dir / f"{case['id']}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", str(out)], check=True)
    ref = {"id": case["id"], "lead_id": case.get("lead_id"), "call_date": case["call_date"], "turns": turns,
           "gold": case["gold"]}
    (out_dir / f"{case['id']}.ref.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{out.name}: {t:.1f}s, {len(turns)} turns")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*")
    args = parser.parse_args()
    out_dir = ROOT / "data" / "demo_audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    for line in (ROOT / "data/eval/calls.jsonl").read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        if looks_like_whatsapp(case["text"]) or (args.ids and case["id"] not in args.ids):
            continue
        await build(case, out_dir)


if __name__ == "__main__":
    asyncio.run(main())
