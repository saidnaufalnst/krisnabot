from __future__ import annotations

from google import genai

from app.core.config import settings


_client: genai.Client | None = None


def get_gemini_client(kind: str = "chat") -> genai.Client:
    if kind != "chat":
        raise ValueError(f"Unsupported Gemini client kind: {kind}")

    global _client
    if _client is None:
        api_key = settings.gemini_api_key
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Set it in .env")
        _client = genai.Client(api_key=api_key)
    return _client
