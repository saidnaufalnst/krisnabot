from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_bool(value: str, default: bool = False) -> bool:
    normalized = (value or "").strip().casefold()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "KRISNABOT")
    environment: str = os.getenv("APP_ENV", "development")
    reload: bool = _parse_bool(os.getenv("APP_RELOAD", "false"))
    host: str = os.getenv("APP_HOST", "127.0.0.1")
    port: int = int(os.getenv("APP_PORT", "8000"))
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/krisnabot",
    )
    database_echo: bool = os.getenv("DATABASE_ECHO", "false").lower() == "true"
    cors_origins: tuple[str, ...] = tuple(
        _parse_csv(
            os.getenv(
                "CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000,http://localhost:9999,http://127.0.0.1:9999",
            )
        )
    )
    cors_origin_regex: str = os.getenv("CORS_ORIGIN_REGEX", "")

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_chat_api_key: str = os.getenv("GEMINI_CHAT_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    model_name: str = os.getenv("MODEL_NAME", "gemini-2.5-flash")
    file_search_model_name: str = os.getenv("FILE_SEARCH_MODEL_NAME", os.getenv("MODEL_NAME", "gemini-2.5-flash"))
    file_search_fallback_model_name: str = os.getenv("FILE_SEARCH_FALLBACK_MODEL_NAME", "")
    file_search_store_name: str = os.getenv("FILE_SEARCH_STORE_NAME", "")
    file_search_store_display_name: str = os.getenv("FILE_SEARCH_STORE_DISPLAY_NAME", "krisnabot-store")
    file_search_top_k: int = int(os.getenv("FILE_SEARCH_TOP_K", "8"))
    file_search_poll_interval_seconds: float = float(os.getenv("FILE_SEARCH_POLL_INTERVAL_SECONDS", "2"))
    file_search_operation_timeout_seconds: float = float(os.getenv("FILE_SEARCH_OPERATION_TIMEOUT_SECONDS", "300"))
    chat_max_output_tokens: int = int(os.getenv("CHAT_MAX_OUTPUT_TOKENS", "400"))
    chat_retry_on_empty_answer: bool = _parse_bool(os.getenv("CHAT_RETRY_ON_EMPTY_ANSWER", "true"))
    technical_help_contact: str = os.getenv("TECHNICAL_HELP_CONTACT", "admin/helpdesk KRISNA di instansi Anda")


settings = Settings()
