"""Speaker diarization and broker/client role assignment.

Two stages:
1. Who spoke when: pyannote.audio (production) or a heuristic fallback that splits on turn gaps
   and clusters segments by simple spectral features (2 speakers: one broker, one client).
2. Which label is the broker: a scoring rule over what each speaker says. Brokers introduce the
   company, offer units, quote prices and payment plans; clients state budgets and needs.
   For a stereo telephony recording each channel is one leg, so roles come from the channel instead.
"""

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.audio.preprocess import SAMPLE_RATE, read_wav
from app.config import get_settings
from app.domain import Segment, Speaker, Transcript
from app.text.normalize import normalize_search

log = logging.getLogger(__name__)

Turn = tuple[float, float, str]  # start, end, label

BROKER_CUES = [
    "معاك من", "معاكي من", "بكلم حضرتك من", "شركة", "ناوي", "nawy", "عندنا", "اقدر اعرض", "اعرض عليك",
    "هبعتلك", "هبعت لحضرتك", "اللوكيشن", "المطور", "الديفلوبر", "ميعاد معاينة", "نحجز", "الوحدة المتاحة",
    "خصم", "نظام السداد", "حضرتك", "يا فندم", "يافندم",
]
CLIENT_CUES = ["ميزانيتي", "انا عايز", "انا عاوز", "بدور على", "محتاج", "معايا", "مراتي", "ولادي", "اقدر ادفع", "غالي"]


def _cue_score(text: str, cues: list[str]) -> int:
    norm = normalize_search(text)
    return sum(norm.count(normalize_search(c)) for c in cues)


@lru_cache
def _pyannote_pipeline():

    from pyannote.audio import Pipeline

    settings = get_settings()
    pipeline = Pipeline.from_pretrained(settings.pyannote_model, use_auth_token=settings.hf_token or None)
    if pipeline is None:  # pyannote returns None (not an error) for a gated model; raising keeps it out of the cache
        raise RuntimeError(f"{settings.pyannote_model}: HF_TOKEN missing or model terms not accepted")
    return pipeline


def pyannote_turns(wav: Path, num_speakers: int | None = 2) -> list[Turn]:
    annotation = _pyannote_pipeline()(str(wav), num_speakers=num_speakers)
    return [(seg.start, seg.end, label) for seg, _, label in annotation.itertracks(yield_label=True)]


def _pitch(clip: np.ndarray, frame: int = 640) -> float:
    """Median autocorrelation pitch (Hz) over voiced 40 ms frames; separates most broker/client pairs."""
    pitches = []
    for i in range(0, len(clip) - frame, frame):
        x = clip[i : i + frame] - clip[i : i + frame].mean()
        if np.sqrt((x**2).mean()) < 0.02:
            continue
        ac = np.correlate(x, x, mode="full")[frame - 1 :]
        lo, hi = SAMPLE_RATE // 400, SAMPLE_RATE // 60  # 60-400 Hz
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.3 * ac[0]:
            pitches.append(SAMPLE_RATE / lag)
    return float(np.median(pitches)) if pitches else 0.0


def _features(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    """Speaker features: log band energies (spectral envelope), zero-crossing rate and median pitch."""
    clip = audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
    if len(clip) < 400:
        return np.zeros(18)
    spec = np.abs(np.fft.rfft(clip * np.hanning(len(clip))))
    bands = np.array_split(spec[: len(spec) // 2], 16)
    energies = np.log(np.array([b.mean() for b in bands]) + 1e-8)
    zcr = np.mean(np.abs(np.diff(np.sign(clip)))) / 2
    return np.concatenate([energies - energies.mean(), [zcr, _pitch(clip)]])


def heuristic_turns(wav: Path, segments: list[Segment]) -> list[Turn]:
    """Two-speaker clustering over ASR segments (k-means, k=2, on standardized features; pitch weighted up)."""
    if not segments:
        return []
    if len(segments) == 1:
        return [(segments[0].start, segments[0].end, "SPEAKER_00")]
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    audio = read_wav(wav)
    feats = StandardScaler().fit_transform(np.array([_features(audio, s.start, s.end) for s in segments]))
    feats[:, -1] *= 3.0  # pitch is the most reliable cue on phone audio; the envelope bands are noisy
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(feats)
    return [(s.start, s.end, f"SPEAKER_{int(lab):02d}") for s, lab in zip(segments, labels, strict=True)]


def assign_turns(transcript: Transcript, turns: list[Turn]) -> Transcript:
    """Label each ASR segment with the diarization turn it overlaps most (word-level when available)."""
    if not turns:
        return transcript
    for seg in transcript.segments:
        best, best_overlap = None, 0.0
        for start, end, label in turns:
            overlap = min(seg.end, end) - max(seg.start, start)
            if overlap > best_overlap:
                best, best_overlap = label, overlap
        if best is None:  # no overlap: nearest turn
            best = min(turns, key=lambda t: min(abs(t[0] - seg.start), abs(t[1] - seg.end)))[2]
        seg.speaker_raw = best
    return transcript


def assign_roles(transcript: Transcript, broker_label: str | None = None) -> Transcript:
    """Map raw diarization labels to broker/client by what each speaker says."""
    labels = sorted({s.speaker_raw for s in transcript.segments if s.speaker_raw})
    if not labels:
        return transcript
    if broker_label is None:
        scores = {}
        for label in labels:
            text = " ".join(s.text for s in transcript.segments if s.speaker_raw == label)
            first_speaker_bonus = 1 if transcript.segments[0].speaker_raw == label else 0  # outbound calls
            scores[label] = _cue_score(text, BROKER_CUES) - _cue_score(text, CLIENT_CUES) + first_speaker_bonus
        broker_label = max(scores, key=scores.__getitem__)
    for s in transcript.segments:
        if s.speaker_raw is None:
            continue
        s.speaker = Speaker.BROKER if s.speaker_raw == broker_label else Speaker.CLIENT
    return transcript


def diarize(wav: Path, transcript: Transcript) -> Transcript:
    backend = get_settings().diarizer
    turns: list[Turn] = []
    if backend == "pyannote":
        try:
            turns = pyannote_turns(wav)
        except Exception:  # model download/auth problems should not lose the call
            log.exception("pyannote failed; falling back to heuristic diarization")
    if not turns:
        turns = heuristic_turns(wav, transcript.segments)
    return assign_roles(assign_turns(transcript, turns))

