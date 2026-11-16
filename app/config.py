from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BC_", env_file=".env", extra="ignore")

    env: str = "local"
    database_url: str = "sqlite:///./data/broker_copilot.db"
    redis_url: str = ""  # empty: Celery runs tasks eagerly in-process
    upload_dir: str = "./data/uploads"
    max_upload_mb: int = 50
    storage_backend: Literal["local", "s3"] = "local"  # s3: AWS S3 / R2 / MinIO, credentials via AWS_* env
    s3_bucket: str = ""
    s3_prefix: str = "uploads"
    s3_endpoint_url: str = ""  # empty = AWS; set for R2 / MinIO
    s3_region: str = ""

    asr_backend: Literal["faster_whisper", "openai", "none"] = "faster_whisper"
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    asr_language: str = "ar"

    diarizer: Literal["pyannote", "heuristic"] = "heuristic"
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    hf_token: str = Field("", validation_alias="HF_TOKEN")

    use_vad: bool = True
    use_noise_reduction: bool = False

    # auto: Claude when Anthropic credentials exist, else Gemini when GOOGLE_API_KEY is set, else rules.
    # An LLM failure falls through the same chain (Claude -> Gemini -> rules).
    summarizer: Literal["auto", "claude", "gemini", "rules"] = "auto"
    claude_model: str = "claude-opus-5-5"
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    gemini_model: str = "gemini-3.5-flash"
    google_api_key: str = Field("", validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"))

    crm_backend: Literal["local", "http"] = "local"
    crm_api_url: str = ""
    crm_api_token: str = ""
    crm_confidence_threshold: float = 0.75

    inventory_path: str = "data/inventory.json"
    dense_model: str = ""  # e.g. BAAI/bge-m3 (needs the `vectors` extra); empty = TF-IDF
    leads_path: str = "data/leads.json"

    whatsapp_verify_token: str = "change-me"
    whatsapp_access_token: str = ""
    whatsapp_graph_url: str = "https://graph.facebook.com/v21.0"
    twilio_auth_token: str = ""

    api_keys: str = "dev:dev-key"  # "actor:key,actor2:key2"

    @property
    def api_key_map(self) -> dict[str, str]:
        pairs = (p.split(":", 1) for p in self.api_keys.split(",") if ":" in p)
        return {key.strip(): actor.strip() for actor, key in pairs}


@lru_cache
def get_settings() -> Settings:
    return Settings()
