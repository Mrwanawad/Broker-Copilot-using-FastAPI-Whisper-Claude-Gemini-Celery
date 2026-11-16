# Checkpoint — Broker Copilot

Last update: 2026-11-16 (session 2: Gemini fallback, synthetic Gemini eval, pyannote install)

## Session 2 (2026-11-16)
- Gemini fallback: `app/summarizer/gemini.py` (google-genai, `response_json_schema=CallSummary`, model
  `gemini-3.5-flash`; 2.5-pro/flash 404 for new users). Shared prompt in `app/summarizer/prompt.py` (stage,
  objection and date rules tightened). Chain: Claude -> Gemini -> rules (`BC_SUMMARIZER` auto|claude|gemini|rules).
  No temperature override (0.2 made Gemini 3 return empty needs intermittently).
- `GOOGLE_API_KEY` / `HF_TOKEN` now loaded via settings aliases (before, non-`BC_` .env keys were ignored).
- Live Gemini eval: 3/5 cases then free-tier 429; key still exhausted. Synthetic replay (labelled):
  `scripts/make_gemini_replay.py` -> `data/eval/gemini_replay.jsonl`, `--replay` flag, report
  `reports/eval_gemini_synthetic.json` (0.85, 1 wrong auto-apply). Suspect gold: c2 areas, c3 follow_up.
- pyannote.audio 3.4 installed; pins torch/torchaudio <2.9, huggingface_hub <1.0 (faster-whisper still OK).
  HF token (user Mrwaan, read) valid but 403 on gated `pyannote/speaker-diarization-3.1` and
  `pyannote/segmentation-3.0`: terms not accepted. Loader now raises (not cached) -> heuristic fallback.
- Object storage: `app/storage.py` (local | s3 via boto3, `s3` extra; R2/MinIO via endpoint). API stores
  refs, pipeline `fetch`es before ASR. render.yaml switched from API disk to S3 env (bucket/keys sync:false).
- Matcher fix: unit type + bedrooms within 1 are now hard filters (UI showed 1BR studios for a 3BR family
  because no 3BR New Cairo unit fits 4.5M x1.15). Call c1 now returns 1 honest match.
- UI checked in Chrome (renders, transcript, CRM diff, objections, matches, draft). Fixed: RTL draft
  (`unicode-bidi: plaintext`, per-call widget key), upload cap 50 MB (`.streamlit/config.toml`).
  RTL fix not visually re-checked (Chrome screenshots timed out).
- Docker: `.dockerignore` added (was shipping .venv/.env), compose `ui` env merge fixed, hf-cache dir
  chowned (worker failed with PermissionError on HF cache). `docker compose up` verified: text + audio
  calls API -> Redis -> Celery -> review on Postgres. Image 2.25 GB, no pyannote inside (heuristic).
- 34 tests, ruff clean.

## Status
All architecture components implemented and running end to end on CPU. 29 tests pass, ruff clean.
Not yet: live Claude eval, pyannote run, Docker build check, browser check of the UI, git repo, deploy.

## Implemented (by area)

### Project setup
- `pyproject.toml` (Python 3.12, uv; extras: `speech`, `arabic`, `vectors`, `ui`, `openai`; ruff config).
- `.env.example` (all `BC_*` settings), `.gitignore`, `.venv` with core deps + faster-whisper, streamlit, edge-tts.
- `app/config.py`: pydantic-settings, `BC_` prefix (ASR backend, diarizer, summarizer mode, Claude model/effort,
  CRM backend + confidence threshold 0.75, DB/Redis URLs, webhook tokens, API keys, optional dense model).
- `app/domain.py`: shared contracts — Segment/Transcript, Evidence, Field_ (value + confidence + evidence),
  Needs, Objection, Commitment, CallSummary, CRMChange (auto_apply / needs_review / unchanged), UnitMatch.

### Audio + ASR + diarization
- `app/audio/preprocess.py`: ffmpeg -> 16 kHz mono, highpass + loudnorm; silero-vad (energy-gate fallback);
  optional noisereduce. Each optional step skips cleanly if its package is missing.
- `app/asr/engines.py`: faster-whisper (CPU int8, word timestamps, VAD filter, real-estate domain prompt,
  no condition-on-previous-text) and OpenAI Whisper API backend. Feeds the decoded numpy array to
  faster-whisper (fixes PyAV 19 / faster-whisper 1.2.1 `metadata_errors` crash).
- `app/diarization/diarize.py`: pyannote 3.1 (HF_TOKEN) with fallback to 2-speaker KMeans on standardized
  features (16 log band energies, zero-crossing rate, autocorrelation pitch weighted x3). Segment-to-turn
  overlap assignment. Broker/client role assignment from speech cues (company intro, offers, prices vs
  budget/needs) + first-speaker bonus.

### Text inputs
- `app/channels/text_threads.py`: WhatsApp export parser (Android + iOS formats, multi-line messages, media/
  deleted system lines dropped, broker detected by name), `broker:/client:` line parser, format detection.

### Arabic / Egyptian text layer (`app/text/`)
- `normalize.py`: display cleanup (diacritics, tatweel, Arabic-Indic digits, punctuation) and search folding
  (alef/ya/teh-marbuta); uses camel-tools when installed.
- `numbers.py`: Egyptian money parsing ("٣ مليون ونص", "مليونين", "خمسمية ألف", "7.5M", "تلاتة مليون و ٢٠٠ ألف",
  "4,250,000"), percentages, years, bedrooms (incl. "أوضتين" and ASR spelling "اوضطين"), amount canonicalization,
  EGP formatting.
- `dates.py`: relative dates (بكرة, بعد بكرة, الأسبوع الجاي, كمان 3 أيام, weekday names -> next occurrence).
- `gazetteer.py`: areas (New Cairo, Sheikh Zayed, 6th of October, New Capital, North Coast, ...), unit types,
  purposes, delivery; Arabic + English aliases + common ASR spellings (شليه); returned in spoken order.

### Summarizer (`app/summarizer/`)
- `claude.py`: Claude structured output via `client.messages.parse(output_format=CallSummary)`, model
  `claude-opus-5-5`, adaptive thinking, effort from settings, segment-indexed transcript + existing CRM record
  in the prompt, refusal / max_tokens handling, token usage returned. Credentials check (API key, auth token,
  `ant` profile).
- `rules.py`: offline deterministic extractor (needs, objections, commitments with resolved dates, sentiment,
  stage, follow-up date, next action, headline), each value with segment + quote evidence. Broker-stated area
  used only when the client named none.
- `playbook.py`: objection categories + cues + suggested responses (price, down payment, installment,
  delivery, location, developer, trust, timing, family decision), commitment / sentiment / payment cues.
- `service.py`: picks Claude or rules (falls back to rules on any Claude error, records the reason); evidence
  verification — quote must be in the cited segment (exact, or near-verbatim for quotes >= 12 chars with every
  number present); unsupported values get confidence x0.4 and are flagged; fills missing suggested responses.

### CRM (`app/crm/`)
- `mapping.py`: summary -> CRM field diff. Policy: confident + empty field = auto_apply; low confidence or
  overwrite of stored value = needs_review; same value = unchanged. Areas merged; activity-log fields
  (last_call_summary, open_objections) always written. `apply_changes` with broker approvals + overrides.
- `client.py`: LocalCRM (DB mirror) and HttpCRM (PATCH to external CRM, then mirror).

### Matching + follow-up (`app/matching/`)
- `matcher.py`: hybrid retrieval — BM25 over Arabic-folded unit text + dense leg (sentence-transformers if
  `BC_DENSE_MODEL` set, else char n-gram TF-IDF), reciprocal rank fusion, then constraint score (area, type,
  bedrooms, budget fit, down payment, plan length, delivery). Hard filters: available, price <= 115% budget.
  Reasons per match.
- `followup.py`: Egyptian-Arabic WhatsApp draft with top units, objection-aware lines, promised callback date.
  Draft only, never auto-sent.

### Storage, queue, pipeline
- `app/db/models.py`: Lead (CRM mirror, JSON fields), Call (status queued/processing/review/approved/failed,
  transcript, summary, CRM diff, flagged fields, matches, draft, per-stage timings, usage/cost, approver),
  FieldCorrection (per reviewed field: auto/accepted/corrected/rejected).
- `app/db/session.py`: SQLite locally, Postgres (psycopg) in Docker/Render, seeds leads from JSON.
- `app/jobs/worker.py`: Celery (acks_late, prefetch 1, time limits); eager in-process when no Redis.
- `app/pipeline.py`: full per-call orchestration with stage timings and cost estimate (Claude tokens,
  Whisper API minutes); failures recorded on the call.

### API (`app/api/`, `app/main.py`)
- X-API-Key auth (actor recorded on approvals).
- `POST /api/calls/audio` (consent required, type/size checks), `POST /api/calls/text`, `GET /api/calls`,
  `GET /api/calls/{id}`, `POST /api/calls/{id}/process`, `POST /api/calls/{id}/approve`
  (approved / rejected / overrides, writes CRM, logs every reviewed field), `GET /api/leads`,
  `GET /api/leads/{id}`, `POST /api/match`, `GET /api/metrics` (calls by status, failure rate, processing s
  per audio minute, queue latency, cost per call, LLM fallback count, correction rate per field), `GET /health`.
- Webhooks: `POST /webhooks/twilio/recording` (HMAC-SHA1 signature check, dedupe by CallSid, lead by phone),
  `GET/POST /webhooks/whatsapp` (verify token; voice notes -> audio call via Graph media download; text
  messages appended to the client's open daily thread).

### UI
- `ui/review_app.py` (Streamlit): submit audio (consent checkbox) or text, call list, RTL transcript with
  speaker labels, CRM diff with confidence, evidence tooltip, approve / reject / edit, objections,
  commitments, matched units table, editable follow-up draft, metrics panel. GFS Didot + Times New Roman.

### Data, evaluation, scripts
- `scripts/make_demo_data.py`: 120 synthetic units (7 areas, plausible 2026 EGP prices) + 3 CRM leads.
- `data/eval/calls.jsonl`: 5 hand-labelled conversations (4 calls + 1 WhatsApp export) with gold fields.
- `app/evaluation/metrics.py`: WER/CER (jiwer, on normalized text), DER (pyannote.metrics or own frame-level
  with collar + optimal label mapping), field accuracy (money within 2%).
- `app/evaluation/run.py`: text eval — per-field accuracy, auto-applied vs wrong auto-applied, latency.
- `scripts/make_demo_audio.py`: two-voice Egyptian TTS calls (edge-tts ar-EG Salma/Shakir) + reference turns.
- `scripts/eval_audio.py`: audio E2E eval — WER/CER, DER, role accuracy, fields from audio, speed.
- `scripts/finetune_whisper.py`: Whisper LoRA fine-tuning (split by call, WER/CER metric, merge, CT2 convert
  notes, 4 GB GPU settings). Not run: no real Egyptian call data.

### Tests (29, `pytest`)
- `test_text.py`: money/percent/years/bedrooms parsing, canonicalization, normalization, gazetteer, dates.
- `test_pipeline_units.py`: WhatsApp parsing, rules extraction, evidence penalty, CRM policy, matcher +
  draft, diarization roles, WER/DER/field metrics.
- `test_api.py`: auth, text call -> review -> approve -> lead updated -> metrics, consent required,
  WhatsApp webhook threading + processing, Twilio signature.
- `test_claude_path.py`: Claude wiring with stubbed client (request shape, evidence verification,
  usage), fallback to rules on error and on refusal.

### DevOps + docs
- `Dockerfile` (ffmpeg, uv, non-root), `docker-compose.yml` (pgvector Postgres, Redis, API, Celery worker,
  Streamlit), `render.yaml` (API + worker + Redis + Postgres, shared env group), `.github/workflows/ci.yml`
  (ruff, pytest, rules eval, docker build).
- `README.md` (pipeline, module map, run, API table, evaluation, guardrails, deploy).
- `../projects-overview.md` row 7 set to IN PROGRESS.

## Results (synthetic data; tuned on the same calls, so optimistic)
- Text -> CRM fields (rules): 0.95 field accuracy, 26 auto-applied, 0 wrong auto-applied.
- Audio (whisper-small CPU int8, heuristic diarization, rules): WER 0.36, CER 0.09, DER 0.15 (0.28 before
  pitch feature), role accuracy 1.00 (was 0.84), fields from audio 0.86 (was 0.65 before ASR-spelling
  aliases), ~18 s processing per audio minute on i7-11370H.
- Reports: `reports/eval_rules.json`, `reports/eval_audio_small.json`.

## Fixes during the session
- PyAV/faster-whisper incompatibility -> pass numpy audio.
- Evidence check too lax on short quotes -> length + numbers rule.
- Correction log missed overrides on unchanged fields -> logged.
- Gold follow-up dates corrected (weekday math); follow-up uses commitments after the call date.
- edge-tts dropped requests -> retry with backoff.

## Open
- Claude summarizer eval (needs ANTHROPIC_API_KEY): `python -m app.evaluation.run --summarizer claude`.
  Server-side refusal fallbacks not enabled yet.
- pyannote diarization: BLOCKED on accepting terms for the two gated repos on HF account Mrwaan; then
  `python scripts/eval_audio.py --diarizer pyannote --whisper-model small`, DER vs heuristic 0.15.
- Live Gemini eval for all 5 cases when quota returns; fix the two suspect gold labels (asked user).
- Bigger held-out eval set; real consented call audio for WER and Whisper fine-tuning.
- Webhooks tested only with stubs.
- S3 path tested with a stubbed client only (no bucket yet).
- pgvector persistence for unit embeddings (in-memory index today).
- No git repo / GitHub remote, no deploy, no demo GIF.
