"""ASR backends behind one interface.

- faster_whisper: local CPU (int8). `small`/`medium` for the demo; large-v3 or an Egyptian-Arabic
  fine-tune (CTranslate2-converted) in production on GPU.
- openai: hosted Whisper API, for when local CPU is too slow.

Each returns a Transcript with segment timestamps (and word timestamps when available, used to
align diarization turns precisely).
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from app.config import get_settings
from app.domain import Segment, Transcript, Word
from app.text.normalize import clean_display

log = logging.getLogger(__name__)

# Domain vocabulary nudges Whisper toward correct spelling of compounds/areas/payment terms.
INITIAL_PROMPT = (
    "مكالمة بين بروكر عقارات وعميل. التجمع الخامس، الشيخ زايد، العاصمة الإدارية، الساحل الشمالي، "
    "مقدم، أقساط، تقسيط على 8 سنين، استلام فوري، تاون هاوس، توين هاوس، فيلا، شقة، مليون جنيه."
)


class ASREngine(Protocol):
    name: str

    def transcribe(self, wav_path: Path, language: str | None = "ar") -> Transcript: ...


class FasterWhisperEngine:
    def __init__(self, model_size: str, device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        self.name = f"faster-whisper:{model_size}"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, wav_path: Path, language: str | None = "ar") -> Transcript:
        from app.audio.preprocess import read_wav

        # pass the decoded 16 kHz array: skips faster-whisper's PyAV decode (and its version pinning)
        segments, info = self.model.transcribe(
            read_wav(wav_path),
            language=language,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
            initial_prompt=INITIAL_PROMPT,
            condition_on_previous_text=False,  # avoids repetition loops on long calls
        )
        out = []
        for s in segments:
            text = clean_display(s.text)
            if not text:
                continue
            words = [Word(start=w.start, end=w.end, text=w.word.strip()) for w in (s.words or [])]
            conf = float(min(1.0, max(0.0, 1.0 + s.avg_logprob))) if s.avg_logprob is not None else None
            out.append(Segment(start=s.start, end=s.end, text=text, words=words, confidence=conf))
        return Transcript(language=info.language, duration_s=info.duration, segments=out, asr_model=self.name)


class OpenAIWhisperEngine:
    name = "openai:whisper-1"

    def transcribe(self, wav_path: Path, language: str | None = "ar") -> Transcript:
        from openai import OpenAI

        client = OpenAI()
        with wav_path.open("rb") as f:
            res = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=language,
                prompt=INITIAL_PROMPT,
                response_format="verbose_json",
                timestamp_granularities=["segment", "word"],
            )
        words = [Word(start=w.start, end=w.end, text=w.word) for w in (getattr(res, "words", None) or [])]
        segs = []
        for s in res.segments or []:
            seg_words = [w for w in words if s.start <= w.start < s.end]
            segs.append(Segment(start=s.start, end=s.end, text=clean_display(s.text), words=seg_words))
        return Transcript(language=res.language or "ar", duration_s=res.duration or 0.0, segments=segs,
                          asr_model=self.name)


@lru_cache
def get_engine() -> ASREngine:
    s = get_settings()
    if s.asr_backend == "openai":
        return OpenAIWhisperEngine()
    if s.asr_backend == "faster_whisper":
        return FasterWhisperEngine(s.whisper_model, s.whisper_device, s.whisper_compute_type)
    raise RuntimeError("ASR disabled (BC_ASR_BACKEND=none): only text inputs are accepted")
