# Broker Copilot (Calls & WhatsApp) — Architecture

## Scenario
Brokers spend hours after calls writing notes and updating the CRM, and much of what clients say is never recorded. The copilot transcribes Egyptian-Arabic calls and WhatsApp voice notes, separates speakers, summarizes needs and objections, auto-fills CRM fields, and suggests matching units and the next action.

## Input
- Call recordings, WhatsApp voice notes and text threads (with recording consent).
- CRM lead record.
- Inventory database.

## Output
- Transcript with speaker labels and timestamps.
- Structured summary JSON: needs (area, type, budget, plan), objections, commitments, sentiment, next action, follow-up date.
- Auto-filled CRM fields (with confidence; low-confidence fields flagged for broker confirmation).
- Matched units and suggested follow-up message draft.

## Architecture

```
 Audio / text (upload, telephony webhook, WhatsApp Business API)
        |
        v
 [Audio pre-processing]  ffmpeg resample, VAD (silero-vad), noise reduction
        |
        v
 [ASR]  Whisper large-v3 (Egyptian Arabic fine-tune) or Whisper API; faster-whisper for CPU
        |
        v
 [Diarization]  pyannote.audio -> broker vs. client
        |
        v
 [Post-processing]  camel-tools normalization, number/price normalization
        |
        v
 [Summarizer]  Claude structured output (Pydantic) with quotes as evidence
        |
        v
 [CRM writer]  field mapping + confidence threshold -> CRM API
        |
        v
 [Unit matcher]  hybrid search over inventory
        |
        v
 [UI]  call review page: transcript, summary, CRM diff to approve
        |
 [Queue]  Celery + Redis for async processing
```

## Tools Used
- Audio: ffmpeg, silero-vad, noisereduce
- ASR: Whisper large-v3 / faster-whisper / Whisper API, Hugging Face Transformers (fine-tuning), datasets
- Diarization: pyannote.audio
- Arabic: camel-tools
- LLM: Claude structured output
- Retrieval: pgvector + BM25
- Backend: FastAPI, Celery, Redis, PostgreSQL
- Integrations: CRM API, Twilio / local telephony, WhatsApp Business API
- UI: Streamlit (review page)
- Evaluation: jiwer (WER/CER), pyannote.metrics (DER)
- DevOps: Docker, GitHub Actions, pytest, ruff

## Tools Not Used (and why)
- Recording without consent: legal requirement.
- Auto-writing low-confidence CRM fields: broker confirms.
- GPU-heavy self-hosting on free tier: use faster-whisper small/medium or API in demo.

## Methodologies
- ASR fine-tuning on Egyptian-Arabic call audio (if data available), evidence-quoted summaries, human-confirmed CRM updates.

## Evaluation Metrics
- WER / CER (ASR), DER (diarization).
- Summary faithfulness (human-graded), CRM field accuracy.
- Processing time per minute of audio.
- Business: minutes saved per broker per day, CRM completeness.

## Production Monitoring
- WER on sampled audited calls, broker correction rate per field.
- Queue latency, failed jobs.
- Token and ASR cost per call.

## Deployment Plan
- Render: FastAPI + worker (Docker), Redis, Postgres.
- UI: Streamlit review page (or served by API).
- Demo audio: self-recorded / synthetic calls only.
