# Broker Copilot (Calls & WhatsApp)

Brokers lose hours after every call writing notes and updating the CRM, and most of what the client
said never gets recorded. Broker Copilot takes an Egyptian-Arabic call recording, WhatsApp voice
note or chat thread and returns:

- a transcript with speaker labels (broker / client) and timestamps,
- a structured summary: needs (area, type, bedrooms, budget, down payment, payment plan, delivery,
  purpose), objections with a suggested answer, commitments with resolved due dates, sentiment,
  stage, next action and follow-up date. **Every value cites the segment and quote it came from.**
- a CRM diff: each field with confidence and status. High-confidence values for empty fields are
  written automatically; low-confidence values and anything that overwrites a stored value wait
  for the broker's approval,
- the 3 best-matching units from live inventory and a follow-up WhatsApp draft in Egyptian Arabic.

Design: [architecture.md](architecture.md).

## Pipeline

```
upload / Twilio recording webhook / WhatsApp Cloud API webhook
  -> ffmpeg 16 kHz mono + loudnorm, silero-vad (energy gate fallback), optional noisereduce
  -> ASR: faster-whisper (CPU int8) or Whisper API, domain prompt, word timestamps
  -> diarization: pyannote 3.1, or 2-speaker spectral clustering fallback
  -> role assignment: broker vs client from what each speaker says (company intro, offers, prices)
  -> Arabic normalization (camel-tools when installed) + Egyptian money parsing ("٣ مليون ونص" -> 3,500,000)
  -> summary: Claude structured output (Pydantic), Gemini as fallback, or the offline rules extractor
  -> evidence check: quotes not found in the transcript cut the field's confidence
  -> CRM diff (threshold policy) -> unit matcher (BM25 + dense, RRF, constraint score) -> follow-up draft
  -> review page: broker approves / corrects; every reviewed field is logged for the correction-rate monitor
Celery + Redis run the jobs; with no Redis configured they run inline.
```

| Module | Path |
| --- | --- |
| Audio pre-processing | `app/audio/preprocess.py` |
| ASR backends | `app/asr/engines.py` |
| Diarization + roles | `app/diarization/diarize.py` |
| WhatsApp export / text parsing | `app/channels/text_threads.py` |
| Arabic normalization, numbers, dates, gazetteer | `app/text/` |
| Claude summarizer, Gemini fallback, rules extractor, evidence check | `app/summarizer/` |
| CRM mapping + adapters | `app/crm/` |
| Unit matcher, follow-up draft | `app/matching/` |
| API, webhooks, metrics | `app/api/routes.py` |
| Queue | `app/jobs/worker.py` |
| Review UI | `ui/review_app.py` |
| Evaluation | `app/evaluation/`, `scripts/eval_audio.py` |
| Whisper LoRA fine-tuning | `scripts/finetune_whisper.py` |

## Run locally

```bash
uv venv && uv pip install -r pyproject.toml && uv pip install faster-whisper streamlit pandas
cp .env.example .env            # ANTHROPIC_API_KEY for Claude, GOOGLE_API_KEY for the Gemini fallback; neither = rules extractor
python scripts/make_demo_data.py
uvicorn app.main:app --reload    # http://localhost:8000/docs
streamlit run ui/review_app.py   # review page
```

Full stack (Postgres + pgvector, Redis, worker, UI): `docker compose up --build`.

API (header `X-API-Key`):

| Method | Path | |
| --- | --- | --- |
| POST | `/api/calls/audio` | multipart audio + `lead_id`, `consent=true` (required) |
| POST | `/api/calls/text` | `broker:/client:` lines or a WhatsApp export |
| GET | `/api/calls/{id}` | transcript, summary, CRM diff, matches, draft, timings, cost |
| POST | `/api/calls/{id}/approve` | `approved_fields`, `rejected_fields`, `overrides` |
| POST | `/api/match` | needs -> units (for other agents, e.g. AI OS) |
| GET | `/api/metrics` | failure rate, latency per audio minute, queue latency, cost per call, correction rate per field |
| POST | `/webhooks/twilio/recording` | Twilio recording callback (signature checked) |
| GET/POST | `/webhooks/whatsapp` | WhatsApp Cloud API verify + inbound (voice notes and text) |

## Evaluation

Data is synthetic: 5 hand-labelled conversations (`data/eval/calls.jsonl`) and their two-voice
audio rendered with Edge TTS Egyptian voices (`scripts/make_demo_audio.py`). The rules extractor was
developed against the same 5 calls, so its text score is optimistic; real numbers need audited calls.

```bash
python -m app.evaluation.run --summarizer rules     # or claude
python scripts/eval_audio.py --whisper-model small  # WER/CER, DER, role accuracy, fields from audio
```

| Run | Result |
| --- | --- |
| Text -> CRM fields, rules | 0.95 field accuracy, 26 auto-applied fields, 0 wrong auto-applied |
| Audio, whisper-small CPU int8, heuristic diarization | WER 0.36, CER 0.09, DER 0.15, broker/client role accuracy 1.00, fields from audio 0.86, 18 s processing per audio minute (i7-11370H) |

WER is high and CER low: most errors are spelling drift in dialect words ("شليه" for "شاليه",
"كش" for "كاش"), not wrong words. The gazetteer and cue lists carry the common ASR spellings, and
the speaker-pitch feature took heuristic DER from 0.28 to 0.15. Those fixes were made while looking
at these same 4 calls, so treat the audio numbers as a smoke test, not a benchmark.

The number that matters most is **wrong auto-applied fields**: a wrong value written without review
silently corrupts the CRM. The threshold (`BC_CRM_CONFIDENCE_THRESHOLD`) is tuned against it.

Gemini (`gemini-3.5-flash`, fallback provider), live run on 3 of 5 cases before the free-tier quota hit:
correct on needs, budget, payment and objections; misses lead stage (says "qualified" where a viewing
was agreed or a discount asked). ~15 s per call. Two gold labels look wrong (c2 areas: client said
"also October"; c3 follow-up: "Saturday" said on a Saturday is 11-21, not 11-15).

**Synthetic** Gemini eval (no quota left): `data/eval/gemini_replay.jsonl` holds made-up responses built
by `scripts/make_gemini_replay.py` from the rules output plus the deviations seen live. They are not
model outputs; they only exercise the Gemini -> evidence check -> CRM path offline:
`python -m app.evaluation.run --summarizer gemini --replay data/eval/gemini_replay.jsonl`
(0.85 field accuracy, 1 wrong auto-applied; `reports/eval_gemini_synthetic.json`, `"synthetic": true`).

Not yet measured: the Claude summarizer (no API key on the build machine; its wiring is covered by
stubbed tests in `tests/test_claude_path.py`), pyannote DER, and anything on real phone audio.

## Guardrails

- No processing without recording consent (`consent=true`; the Twilio flow records only after the consent prompt).
- Low-confidence fields and overwrites of stored values are never auto-written; the broker confirms.
- Quotes are checked against the transcript; a value with an invented quote drops below the threshold.
- Follow-up messages are drafts; nothing is sent to the client automatically.
- Demo data is synthetic; no real client audio in the repo.

## Tests

`pytest` (29 tests: number/date parsing, WhatsApp parsing, extraction, evidence check, CRM policy,
matcher, diarization roles, metrics, API review flow, webhooks, Claude path with a stubbed client).
`ruff check app tests scripts ui`. CI: `.github/workflows/ci.yml`.

## Deploy

`render.yaml`: API (Docker) + worker + Redis + Postgres. Uploads go to S3-compatible object storage
(`BC_STORAGE_BACKEND=s3`, `BC_S3_BUCKET`, `BC_S3_ENDPOINT_URL` for R2/MinIO, `AWS_*` credentials), so the
API and the worker scale separately; `Call.source_path` holds `s3://bucket/key` and the worker downloads it
before ASR. Locally the default is disk (`BC_UPLOAD_DIR`).

`docker compose up -d --build` runs Postgres (pgvector), Redis, API (:8000), Celery worker and the review
UI (:8501). Verified 2026-11-16: text and audio calls go API -> Redis -> worker -> review.
