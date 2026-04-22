from __future__ import annotations

import re
import time
from typing import Any

from google.genai import types
from google.genai.errors import ClientError

from app.core.audit_logger import AuditLogger
from app.core.config import settings
from app.repositories.chat_log_repository import ChatLogRepository
from app.services.gemini_client import get_gemini_client
from app.services.gemini_file_search_service import GeminiFileSearchService
from app.services.ingestion_service import IngestionService


SYSTEM_INSTRUCTION = (
    "Anda adalah KRISNABOT, asisten tanya jawab dokumen KRISNA.\n"
    "Jawab dalam bahasa Indonesia hanya dari rujukan File Search.\n"
    "Jika rujukan tidak cukup, katakan bahwa jawabannya belum tersedia di dokumen.\n"
    "Untuk prosedur, berikan langkah singkat dan berurutan.\n"
    "Pakai 'Catatan:' untuk syarat atau batasan penting.\n"
    "Jangan menebak dan jangan memakai markdown code fence."
)

GREETINGS = {
    "halo", "hai", "hi", "pagi", "siang", "sore", "malam", "halo bot", "halo krisnabot"
}
THANKS = {"terima kasih", "makasih", "thanks", "thank you"}
CLOSINGS = {"bye", "dadah", "sampai jumpa", "cukup", "sudah itu saja", "oke cukup"}

KRISNA_SCOPE_HINTS = {
    "krisna", "aplikasi", "sistem", "dokumen", "menu", "modul", "akun", "login",
    "user", "role", "perencanaan", "penganggaran", "kegiatan", "indikator", "output",
    "validasi", "upload", "pdf", "admin", "referensi", "usulan", "anggaran", "sasaran",
    "program", "subkegiatan", "form", "laporan",
}
OUT_OF_SCOPE_HINTS = {
    "cuaca", "film", "musik", "lagu", "resep", "masakan", "presiden", "menteri",
    "sepak bola", "bola", "skor", "bitcoin", "saham", "crypto", "puisi", "cerpen",
    "novel", "translate", "terjemah", "bahasa inggris", "python", "javascript",
    "java", "coding", "programming", "anime", "game", "joke", "candaan", "humor",
}
TECHNICAL_HELP_HINTS = {
    "error", "gagal", "tidak bisa", "ga bisa", "gabisa", "tidak dapat", "tidak muncul",
    "tidak kebuka", "tidak terbuka", "tidak masuk", "kendala", "masalah", "bermasalah",
    "bug", "loading", "blank", "putih", "timeout", "server", "akses", "login",
    "password", "akun", "sinkron", "sinkronisasi", "jaringan", "403", "404", "500",
    "503", "upload", "ingest", "crash", "hang",
}
RETRYABLE_PROVIDER_ERROR_HINTS = (
    " 429 ", " 500 ", " 503 ", "deadline exceeded", "high demand", "internal error",
    "resource exhausted", "resource_exhausted", "service unavailable",
    "temporarily unavailable", "too many requests", "try again later",
    "unavailable",
)

# TTL (detik) untuk cache ringan di level request
_INDEXED_FILES_CACHE_TTL = 10.0
_STORE_NAMES_CACHE_TTL = 30.0


class RAGService:
    def __init__(self) -> None:
        self.audit_logger = AuditLogger("chat_query")
        self.chat_log_repository = ChatLogRepository()
        self._ingestion: IngestionService | None = None
        self._file_search_service: GeminiFileSearchService | None = None
        self._client = None

        # Cache ringan untuk mengurangi DB hit per request.
        self._indexed_files_cache: list[str] = []
        self._indexed_files_cache_at: float = 0.0
        self._store_names_cache: list[str] = []
        self._store_names_cache_at: float = 0.0

        # Pre-build daftar model sekali saja.
        self._candidate_models: list[str] | None = None

        # Config dibaca sekali.
        self._max_output_tokens: int = max(int(getattr(settings, "chat_max_output_tokens", 400) or 400), 300)
        self._top_k: int = max(int(getattr(settings, "file_search_top_k", 8) or 8), 1)

    # ------------------------------------------------------------------
    # Properties (lazy init, satu instance per RAGService)
    # ------------------------------------------------------------------

    @property
    def ingestion(self) -> IngestionService:
        if self._ingestion is None:
            self._ingestion = IngestionService()
        return self._ingestion

    @ingestion.setter
    def ingestion(self, service: IngestionService) -> None:
        self._ingestion = service

    @property
    def file_search_service(self) -> GeminiFileSearchService:
        if self._file_search_service is None:
            self._file_search_service = GeminiFileSearchService()
        return self._file_search_service

    @file_search_service.setter
    def file_search_service(self, service: GeminiFileSearchService) -> None:
        self._file_search_service = service

    @property
    def client(self):
        if self._client is None:
            self._client = get_gemini_client(kind="chat")
        return self._client

    @client.setter
    def client(self, value) -> None:
        self._client = value

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_indexed_files(self) -> list[str]:
        now = time.monotonic()
        if now - self._indexed_files_cache_at > _INDEXED_FILES_CACHE_TTL:
            self._indexed_files_cache = self.ingestion.list_indexed_files()
            self._indexed_files_cache_at = now
        return self._indexed_files_cache

    def _get_store_names(self) -> list[str]:
        now = time.monotonic()
        if now - self._store_names_cache_at > _STORE_NAMES_CACHE_TTL:
            names = self.ingestion.list_file_search_store_names()
            if not names:
                names = [self.file_search_service.get_or_create_store_name()]
            self._store_names_cache = names
            self._store_names_cache_at = now
        return self._store_names_cache

    def invalidate_cache(self) -> None:
        """Panggil ini setelah ingest selesai agar cache langsung stale."""
        self._indexed_files_cache_at = 0.0
        self._store_names_cache_at = 0.0

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_chat(
        self,
        question: str,
        answer: str,
        found: bool,
        used_files: list[str],
        error: str = "",
    ) -> None:
        try:
            self.chat_log_repository.create(
                question=question,
                answer=answer,
                found=found,
                used_files=used_files,
            )
        except Exception:
            pass
        try:
            self.audit_logger.write(
                {"question": question, "found": found, "used_files": used_files, "error": error}
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").casefold().strip().split())

    @staticmethod
    def _strip_code_fences(text: str | None) -> str:
        if not text:
            return ""
        result = text.strip()
        if result.startswith("```"):
            result = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", result)
            result = re.sub(r"\s*```$", "", result).strip()
        return result

    @staticmethod
    def _contains_any(text: str, phrases: set[str]) -> bool:
        return any(phrase in text for phrase in phrases)

    # ------------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------------

    def _detect_social_response(self, question: str) -> dict[str, Any] | None:
        normalized = self._normalize_text(question)
        if not normalized:
            return None
        if normalized in GREETINGS:
            return {
                "found": True,
                "answer": "Halo, saya KRISNABOT. Silakan ajukan pertanyaan terkait KRISNA.",
                "message": "", "error": "", "used_files": [],
            }
        if normalized in THANKS:
            return {
                "found": True,
                "answer": "Sama-sama. Silakan lanjutkan jika masih ada yang ingin ditanyakan.",
                "message": "", "error": "", "used_files": [],
            }
        if normalized in CLOSINGS:
            return {
                "found": True,
                "answer": "Baik, sampai jumpa. Saya siap membantu lagi jika diperlukan.",
                "message": "", "error": "", "used_files": [],
            }
        return None

    def _is_out_of_scope_question(self, question: str) -> bool:
        normalized = self._normalize_text(question)
        if not normalized or "krisna" in normalized:
            return False
        return self._contains_any(normalized, OUT_OF_SCOPE_HINTS) and not self._contains_any(
            normalized, KRISNA_SCOPE_HINTS
        )

    def _needs_technical_help(self, question: str) -> bool:
        normalized = self._normalize_text(question)
        return bool(normalized) and self._contains_any(normalized, TECHNICAL_HELP_HINTS)

    # ------------------------------------------------------------------
    # Response builders
    # ------------------------------------------------------------------

    @staticmethod
    def _technical_help_target() -> str:
        return str(getattr(settings, "technical_help_contact", "") or "admin/helpdesk KRISNA di instansi Anda").strip()

    @staticmethod
    def _is_model_not_found_error(message: str, model_name: str) -> bool:
        lowered = (message or "").casefold()
        return "not found" in lowered and model_name.casefold() in lowered

    @staticmethod
    def _is_retryable_provider_error(message: str) -> bool:
        lowered = f" {(message or '').casefold()} "
        return any(hint in lowered for hint in RETRYABLE_PROVIDER_ERROR_HINTS)

    @staticmethod
    def _build_out_of_scope_response() -> dict[str, Any]:
        return {
            "found": False, "answer": "",
            "message": (
                "Saya hanya menjawab pertanyaan terkait KRISNA. "
                "Silakan tanyakan proses, menu, aturan, dokumen, atau kendala penggunaan KRISNA."
            ),
            "error": "out_of_scope_question", "used_files": [],
        }

    def _build_technical_help_response(self, prefix: str = "") -> dict[str, Any]:
        message = (
            "Saya belum menemukan rujukan yang cukup untuk kendala teknis tersebut. "
            f"Silakan hubungi {self._technical_help_target()} dengan menyertakan nama fitur, waktu kejadian, "
            "pesan error, dan screenshot jika ada."
        )
        if prefix:
            message = f"{prefix.rstrip()} {message}"
        return {
            "found": False, "answer": "",
            "message": message,
            "error": "technical_help_required", "used_files": [],
        }

    def _build_provider_unavailable_response(self) -> dict[str, Any]:
        return {
            "found": False, "answer": "",
            "message": (
                "Layanan Gemini sedang sibuk. "
                f"Silakan coba lagi beberapa saat. Jika kendala mendesak, hubungi {self._technical_help_target()}."
            ),
            "error": "provider_unavailable", "used_files": [],
        }

    # ------------------------------------------------------------------
    # Model list (di-build sekali, di-cache di instance)
    # ------------------------------------------------------------------

    def _get_candidate_models(self) -> list[str]:
        if self._candidate_models is None:
            models: list[str] = []
            for candidate in (
                getattr(settings, "file_search_model_name", ""),
                getattr(settings, "file_search_fallback_model_name", ""),
                getattr(settings, "model_name", ""),
            ):
                model_name = str(candidate or "").strip()
                if model_name and model_name not in models:
                    models.append(model_name)
            self._candidate_models = models
        return self._candidate_models

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_file_search_prompt(question: str) -> str:
        return (
            "Jawab pertanyaan ini berdasarkan dokumen KRISNA dari File Search.\n"
            "Jika ada prosedur, tulis langkah singkat dan berurutan.\n"
            f"Pertanyaan: {question}"
        )

    # ------------------------------------------------------------------
    # Core API call
    # ------------------------------------------------------------------

    def _call_file_search_model(self, prompt: str, model_name: str):
        store_names = self._get_store_names()
        return self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.1,
                max_output_tokens=self._max_output_tokens,
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=store_names,
                            top_k=self._top_k,
                        )
                    )
                ],
            ),
        )

    def _parse_file_search_response(self, response: Any) -> dict[str, Any]:
        answer = self._strip_code_fences(str(getattr(response, "text", "") or "").strip())
        used_files = self.file_search_service.extract_used_files(response)
        has_grounding = self.file_search_service.has_grounding(response)

        if not has_grounding:
            return {
                "found": False, "answer": "",
                "message": (
                    "Saya belum menemukan rujukan yang cukup relevan di dokumen. "
                    "Coba gunakan istilah KRISNA yang lebih spesifik."
                ),
                "error": "context_not_found", "used_files": [],
            }
        if not answer:
            return {
                "found": False, "answer": "",
                "message": (
                    "Saya belum menemukan jawaban yang cukup jelas dari dokumen. "
                    "Coba gunakan pertanyaan yang lebih spesifik."
                ),
                "error": "empty_model_text", "used_files": used_files,
            }
        return {"found": True, "answer": answer, "message": "", "error": "", "used_files": used_files}

    def _generate_answer_with_file_search(self, prompt: str) -> dict[str, Any]:
        candidate_models = self._get_candidate_models()
        retryable_error_seen = False
        model_not_found_models: list[str] = []
        last_message = ""

        for model_name in candidate_models:
            try:
                response = self._call_file_search_model(prompt, model_name)
                return self._parse_file_search_response(response)
            except ClientError as exc:
                message = str(exc)
                last_message = message
                if self._is_model_not_found_error(message, model_name):
                    model_not_found_models.append(model_name)
                    continue
                if self._is_retryable_provider_error(message):
                    retryable_error_seen = True
                    continue
                return {
                    "found": False, "answer": "",
                    "message": f"Terjadi kesalahan saat memproses jawaban File Search: {message}",
                    "error": "client_error", "used_files": [],
                }
            except Exception as exc:
                message = str(exc)
                last_message = message
                if self._is_retryable_provider_error(message):
                    retryable_error_seen = True
                    continue
                return {
                    "found": False, "answer": "",
                    "message": f"Terjadi kesalahan tak terduga saat menghasilkan jawaban: {exc}",
                    "error": "unexpected_generation_error", "used_files": [],
                }

        if retryable_error_seen and model_not_found_models:
            return {
                "found": False, "answer": "",
                "message": (
                    "Model utama File Search sedang sibuk dan model cadangan tidak tersedia. "
                    "Silakan coba lagi beberapa saat atau periksa konfigurasi FILE_SEARCH_FALLBACK_MODEL_NAME."
                ),
                "error": "fallback_model_unavailable", "used_files": [],
            }
        if retryable_error_seen:
            return self._build_provider_unavailable_response()
        if model_not_found_models:
            tried = ", ".join(model_not_found_models)
            return {
                "found": False, "answer": "",
                "message": (
                    f"Model Gemini untuk File Search tidak tersedia: {tried}. "
                    "Periksa FILE_SEARCH_MODEL_NAME dan FILE_SEARCH_FALLBACK_MODEL_NAME di konfigurasi."
                ),
                "error": "model_not_found", "used_files": [],
            }
        return {
            "found": False, "answer": "",
            "message": (
                "Terjadi kegagalan saat memproses jawaban File Search."
                if not last_message
                else f"Terjadi kegagalan saat memproses jawaban File Search: {last_message}"
            ),
            "error": "unexpected_generation_error", "used_files": [],
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def ask(self, question: str) -> dict[str, Any]:
        question = (question or "").strip()

        if not question:
            result = {
                "found": False, "answer": "",
                "message": "Pertanyaan tidak boleh kosong.",
                "error": "empty_question", "used_files": [],
            }
            self._log_chat(question, result["message"], False, [], result["error"])
            return result

        # --- Fast-path: social / greeting (no DB, no API) ---
        social = self._detect_social_response(question)
        if social is not None:
            self._log_chat(question, social["answer"], True, [], "")
            return social

        # --- Fast-path: out-of-scope (no DB, no API) ---
        if self._is_out_of_scope_question(question):
            result = self._build_out_of_scope_response()
            self._log_chat(question, result["message"], False, [], result["error"])
            return result

        technical_help_requested = self._needs_technical_help(question)

        # --- Cek indexed files via cache (satu DB query max per TTL) ---
        indexed_files = self._get_indexed_files()
        if not indexed_files:
            if technical_help_requested:
                result = self._build_technical_help_response("Dokumen referensi KRISNA belum tersedia di sistem.")
            else:
                result = {
                    "found": False, "answer": "",
                    "message": (
                        "Dokumen KRISNA belum tersedia untuk dijadikan rujukan. "
                        "Pastikan dokumen sudah di-upload dan di-ingest."
                    ),
                    "error": "context_not_found", "used_files": [],
                }
            self._log_chat(question, result["message"], False, [], result["error"])
            return result

        # --- Panggil Gemini File Search ---
        prompt = self._build_file_search_prompt(question)
        result = self._generate_answer_with_file_search(prompt)

        # --- Retry HANYA jika grounding ditemukan tapi teks kosong ---
        # (lebih konservatif: tidak retry jika context_not_found, karena retry tidak akan membantu)
        if (
            bool(getattr(settings, "chat_retry_on_empty_answer", True))
            and result.get("error") == "empty_model_text"
            and result.get("used_files")
        ):
            result = self._generate_answer_with_file_search(prompt)

        found: bool = bool(result.get("found", False))
        answer: str = str(result.get("answer", "") or "").strip()
        message: str = str(result.get("message", "") or "").strip()
        error: str = str(result.get("error", "") or "").strip()
        used_files: list[str] = list(result.get("used_files") or [])

        # --- Technical help fallback ---
        if technical_help_requested and (not found or not answer) and error not in {
            "provider_unavailable", "fallback_model_unavailable", "model_not_found",
        }:
            technical_result = self._build_technical_help_response(message)
            self._log_chat(
                question=question,
                answer=technical_result["message"],
                found=False,
                used_files=[],
                error=technical_result["error"],
            )
            return technical_result

        final_result = {
            "found": found and bool(answer),
            "answer": answer if found and answer else "",
            "message": (
                ""
                if found and answer
                else (message or "Saya belum menemukan jawaban yang sesuai di dokumen.")
            ),
            "error": error,
            "used_files": used_files if found and answer else [],
        }

        self._log_chat(
            question=question,
            answer=final_result["answer"] or final_result["message"],
            found=final_result["found"],
            used_files=final_result["used_files"],
            error=final_result["error"],
        )
        return final_result
