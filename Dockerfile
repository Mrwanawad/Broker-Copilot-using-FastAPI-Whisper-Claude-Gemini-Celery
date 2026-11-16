FROM python:3.12-slim

# ffmpeg for audio pre-processing; libgomp for CTranslate2 (faster-whisper)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 UV_SYSTEM_PYTHON=1

# speech extras minus pyannote (torch-heavy); enable BC_DIARIZER=pyannote on a GPU image
COPY pyproject.toml ./
RUN uv pip install --no-cache -r pyproject.toml && \
    uv pip install --no-cache "faster-whisper>=1.1" "soundfile>=0.12" "streamlit>=1.40" "pandas>=2.2" "boto3>=1.35"

COPY app ./app
COPY ui ./ui
COPY .streamlit ./.streamlit
COPY scripts ./scripts
COPY data/inventory.json data/leads.json ./data/
COPY data/eval ./data/eval

# cache dir must exist (owned by copilot) before the hf-cache volume mounts over it, or Docker creates it as root
RUN useradd -m copilot && mkdir -p /app/data/uploads /home/copilot/.cache/huggingface \
    && chown -R copilot /app /home/copilot/.cache
USER copilot
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
