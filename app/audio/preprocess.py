"""Audio pre-processing: any input format -> 16 kHz mono PCM WAV, optional VAD trim and noise reduction.

ffmpeg handles every container brokers send (WhatsApp .opus/.ogg, telephony .mp3/.wav, phone .m4a).
silero-vad and noisereduce are optional; each step is skipped cleanly when its package is missing.
"""

import json
import logging
import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


@dataclass
class PreparedAudio:
    path: Path
    duration_s: float
    speech_regions: list[tuple[float, float]] = field(default_factory=list)  # VAD output, seconds
    steps: list[str] = field(default_factory=list)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def probe_duration(path: Path) -> float:
    if shutil.which("ffprobe"):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, check=False,
        )
        try:
            return float(json.loads(out.stdout)["format"]["duration"])
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def to_wav16k(src: Path, dst: Path) -> Path:
    """Resample to 16 kHz mono 16-bit PCM. Loudness-normalized so quiet phone legs are audible to ASR."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-af", "highpass=f=80,loudnorm=I=-20:TP=-2",
        "-c:a", "pcm_s16le", str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def vad_regions(audio: np.ndarray) -> list[tuple[float, float]]:
    """Speech regions via silero-vad; falls back to an energy gate."""
    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad

        model = load_silero_vad()
        stamps = get_speech_timestamps(torch.from_numpy(audio), model, sampling_rate=SAMPLE_RATE)
        return [(s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE) for s in stamps]
    except ImportError:
        return energy_regions(audio)


def energy_regions(audio: np.ndarray, frame_s: float = 0.03, min_gap_s: float = 0.4) -> list[tuple[float, float]]:
    frame = int(SAMPLE_RATE * frame_s)
    if len(audio) < frame:
        return []
    n = len(audio) // frame
    rms = np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    threshold = max(np.percentile(rms, 30) * 2.0, 0.01)
    voiced = rms > threshold
    regions: list[tuple[float, float]] = []
    start = None
    for i, v in enumerate(voiced):
        t = i * frame_s
        if v and start is None:
            start = t
        elif not v and start is not None:
            regions.append((start, t))
            start = None
    if start is not None:
        regions.append((start, n * frame_s))
    merged: list[tuple[float, float]] = []
    for s, e in regions:
        if merged and s - merged[-1][1] < min_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= 0.2]


def denoise(audio: np.ndarray) -> np.ndarray:
    import noisereduce as nr

    return nr.reduce_noise(y=audio, sr=SAMPLE_RATE, stationary=True, prop_decrease=0.8)


def prepare(src: Path, work_dir: Path, use_vad: bool = True, use_noise_reduction: bool = False) -> PreparedAudio:
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found on PATH")
    out = to_wav16k(src, work_dir / f"{src.stem}.16k.wav")
    steps = ["ffmpeg:16k-mono-loudnorm"]
    audio = read_wav(out)
    if use_noise_reduction:
        try:
            audio = denoise(audio)
            write_wav(out, audio)
            steps.append("noisereduce")
        except ImportError:
            log.info("noisereduce not installed; skipping")
    regions = vad_regions(audio) if use_vad else []
    if use_vad:
        steps.append("vad")
    return PreparedAudio(out, len(audio) / SAMPLE_RATE, regions, steps)
