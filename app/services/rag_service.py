from __future__ import annotations

import logging
import re
import time
from typing import Any

from google.genai import types
from google.genai.errors import ClientError, ServerError

from app.core.audit_logger import AuditLogger
from app.core.config import settings
from app.repositories.chat_log_repository import ChatLogRepository
from app.services.gemini_client import get_gemini_client
from app.services.gemini_file_search_service import GeminiFileSearchService
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System instruction
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "Anda adalah KRISNABOT, asisten tanya jawab dokumen KRISNA.\n"
    "Jawab hanya dari rujukan File Search dokumen KRISNA, dalam bahasa Indonesia.\n"
    "Jika rujukan tidak cukup jelas, katakan jawaban belum tersedia di dokumen.\n"
    "Fokus menjawab inti pertanyaan pengguna secara jelas, padat, dan relevan.\n"
    "Jawab tepat pada objek yang ditanyakan pengguna dan jangan mengalihkan ke topik, menu, atau data lain jika tidak diperlukan.\n"
    "Jangan menambahkan asumsi, alternatif maksud, atau informasi tambahan yang tidak diminta pengguna.\n"
    "Jika pertanyaan berupa keluhan atau hambatan, jelaskan langkah yang perlu dicek atau dilakukan berdasarkan dokumen.\n"
    "Jawab dengan detail jika dokumen memuat detail penting.\n"
    "Untuk pertanyaan tata cara, berikan langkah yang runtut, operasional, dan cukup detail, tetapi tetap ringkas.\n"
    "Utamakan jawaban yang langsung membantu pengguna menyelesaikan pertanyaannya.\n"
    "Sebutkan menu, tombol, status, syarat, atau hak akses jika memang disebut di rujukan.\n"
    "Tambahkan 'Catatan:' jika ada informasi atau batasan penting dari rujukan.\n"
    "Jangan pakai markdown code fence, heading, atau citation."
)

GREETINGS = {"halo", "hai", "hi", "pagi", "siang", "sore", "malam", "halo bot", "halo krisnabot"}
THANKS    = {"terima kasih", "makasih", "thanks", "thank you"}
CLOSINGS  = {"bye", "dadah", "sampai jumpa", "cukup", "sudah itu saja", "oke cukup"}
RANDOM_INPUTS = {"tes","test","testing","testing testing","cek","cek cek","ping","p","apa kabar",}
RANDOM_INPUT_WORDS = {"tes", "test", "testing", "cek", "coba", "ping", "p"}

_INDEXED_FILES_TTL = 120.0   # seconds
_STORE_NAMES_TTL   = 600.0


# ---------------------------------------------------------------------------
# RAGService
# ---------------------------------------------------------------------------

class RAGService:
    def __init__(self) -> None:
        self.audit_logger         = AuditLogger("chat_query")
        self.chat_log_repository  = ChatLogRepository()
        self._ingestion: IngestionService | None            = None
        self._file_search_service: GeminiFileSearchService | None = None
        self._client                                        = None

        # Cache
        self._indexed_files_cache: list[str] = []
        self._indexed_files_ts:    float     = 0.0
        self._store_names_cache:   list[str] = []
        self._store_names_ts:      float     = 0.0

        # Config (read once)
        self._max_tokens = max(int(getattr(settings, "chat_max_output_tokens", 500) or 500), 250)
        self._top_k      = max(int(getattr(settings, "file_search_top_k", 5) or 5), 1)
        self._retry_attempts = max(int(getattr(settings, "chat_retry_attempts", 2) or 2), 0)
        self._retry_backoff_seconds = max(
            float(getattr(settings, "chat_retry_backoff_seconds", 1) or 1),
            0.0,
        )

    # ------------------------------------------------------------------
    # Lazy properties
    # ------------------------------------------------------------------

    @property
    def ingestion(self) -> IngestionService:
        if self._ingestion is None:
            self._ingestion = IngestionService()
        return self._ingestion

    @ingestion.setter
    def ingestion(self, svc: IngestionService) -> None:
        self._ingestion = svc

    @property
    def file_search_service(self) -> GeminiFileSearchService:
        if self._file_search_service is None:
            self._file_search_service = GeminiFileSearchService()
        return self._file_search_service

    @file_search_service.setter
    def file_search_service(self, svc: GeminiFileSearchService) -> None:
        self._file_search_service = svc

    @property
    def client(self):
        if self._client is None:
            self._client = get_gemini_client(kind="chat")
        return self._client

    @client.setter
    def client(self, value) -> None:
        self._client = value

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _get_indexed_files(self) -> list[str]:
        now = time.monotonic()
        if now - self._indexed_files_ts > _INDEXED_FILES_TTL:
            try:
                self._indexed_files_cache = self.ingestion.list_indexed_files() or []
            except Exception as exc:
                logger.warning("[RAG] list_indexed_files error: %s", exc)
            self._indexed_files_ts = now
        return self._indexed_files_cache

    def _get_store_names(self) -> list[str]:
        now = time.monotonic()
        if now - self._store_names_ts > _STORE_NAMES_TTL:
            try:
                names = self.ingestion.list_file_search_store_names()
            except Exception:
                names = []
            if not names:
                try:
                    fallback = self.file_search_service.get_or_create_store_name()
                    if fallback:
                        names = [fallback]
                except Exception as exc:
                    logger.error("[RAG] get_or_create_store_name error: %s", exc)
            self._store_names_cache = names
            self._store_names_ts    = now
        return self._store_names_cache

    def invalidate_cache(self) -> None:
        self._indexed_files_ts = 0.0
        self._store_names_ts   = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join((text or "").casefold().strip().split())

    @staticmethod
    def _clean_answer(text: str | None) -> str:
        def normalize_note_label(match: re.Match[str]) -> str:
            label = match.group("label").casefold()
            rest = re.sub(r"^\*\s+", "", match.group("rest") or "").strip()
            normalized_label = "Catatan penting" if label == "catatan penting" else "Catatan"
            return f"**{normalized_label}:**" + (f" {rest}" if rest else "")

        if not text:
            return ""
        result = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if result.startswith("```"):
            result = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", result)
            result = re.sub(r"\s*```$", "", result).strip()
        if "\n\n[" in result:
            prefix, _, suffix = result.partition("\n\n[")
            if "[cite:" in suffix:
                result = prefix.strip()

        cleaned_lines: list[str] = []
        for line in result.split("\n"):
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue
            if stripped.casefold() == "prosedur:":
                continue

            line = re.sub(r"^(informasi|prosedur)\s*:\s*", "", line, flags=re.IGNORECASE)
            line = re.sub(r"^\*\s+", "- ", line)
            line = re.sub(
                r"^(?:\*\*)?(?P<label>catatan penting|catatan)\s*:\s*(?:\*\*)?(?P<rest>.*)$",
                normalize_note_label,
                line,
                flags=re.IGNORECASE,
            )
            cleaned_lines.append(line.rstrip())

        result = "\n".join(cleaned_lines)
        result = re.sub(r"\n{3,}", "\n\n", result).strip()
        result = re.sub(r"\n\n(?:\d+[.)]|[-*])\s*$", "", result).rstrip()

        if re.search(r"\b(?:atau|dan|serta|maupun)\s*$", result, flags=re.IGNORECASE):
            last_sentence_end = max(result.rfind("."), result.rfind("!"), result.rfind("?"))
            if last_sentence_end != -1:
                result = result[: last_sentence_end + 1].rstrip()
        return result

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        msg = str(exc).casefold()
        retryable_hints = (
            "429", "500", "503", "504",
            "resource_exhausted", "unavailable",
            "deadline exceeded", "deadline expired",
            "high demand", "too many requests",
            "service unavailable", "temporarily unavailable",
            "timeout", "timed out",
        )
        return any(h in msg for h in retryable_hints)

    def _technical_help_target(self) -> str:
        return str(getattr(settings, "technical_help_contact", "") or "admin/helpdesk KRISNA di instansi Anda").strip()

    def _model_name(self) -> str:
        return str(getattr(settings, "model_name", "") or "").strip()

    def _helpdesk_message(self) -> str:
        return f"Saya belum bisa menjawab ini. Silakan hubungi {self._technical_help_target()}."

    @staticmethod
    def _looks_like_unanswered_answer(answer: str) -> bool:
        normalized = RAGService._normalize(answer)
        patterns = (
            r"\bbelum tersedia di dokumen\b",
            r"\bbelum menemukan jawaban\b",
            r"\bbelum menemukan rujukan\b",
            r"\btidak ditemukan\b",
            r"\btidak tersedia\b",
        )
        return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _looks_like_random_input(question: str) -> bool:
        normalized = RAGService._normalize(question)
        if not normalized:
            return False
        if normalized in RANDOM_INPUTS:
            return True
        if len(normalized) == 1:
            return True

        words = normalized.split()
        if len(words) <= 3 and all(word in RANDOM_INPUT_WORDS for word in words):
            return True
        return False

    # ------------------------------------------------------------------
    # Response builders
    # ------------------------------------------------------------------

    @staticmethod
    def _ok(
        answer: str,
        used_files: list[str] | None = None,
        grounding_chunks: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        return {
            "found": True,
            "answer": answer,
            "message": "",
            "error": "",
            "used_files": used_files or [],
            "grounding_chunks": grounding_chunks or [],
        }

    @staticmethod
    def _fail(
        message: str,
        error: str,
        used_files: list[str] | None = None,
        grounding_chunks: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        return {
            "found": False,
            "answer": "",
            "message": message,
            "error": error,
            "used_files": used_files or [],
            "grounding_chunks": grounding_chunks or [],
        }

    def _resp_provider_unavailable(self) -> dict[str, Any]:
        return self._fail(
            f"Layanan Gemini sedang sibuk. Coba lagi beberapa saat atau hubungi {self._technical_help_target()}.",
            "provider_unavailable",
        )

    # ------------------------------------------------------------------
    # Social / greeting detection
    # ------------------------------------------------------------------

    def _detect_social(self, question: str) -> dict[str, Any] | None:
        n = self._normalize(question)
        if n in GREETINGS:
            return self._ok("Halo, saya KRISNABOT. Silakan ajukan pertanyaan terkait KRISNA.")
        if self._looks_like_random_input(question):
            return self._ok("Halo, saya KRISNABOT. Silakan ajukan pertanyaan terkait KRISNA.")
        if n in THANKS:
            return self._ok("Sama-sama. Silakan lanjutkan jika masih ada yang ingin ditanyakan.")
        if n in CLOSINGS:
            return self._ok("Baik, sampai jumpa. Saya siap membantu lagi jika diperlukan.")
        return None

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(question: str) -> str:
        """
        Prompt utama — minta model mencari secara semantik, bukan hanya kata kunci persis.
        Ini yang membantu KRISNABOT memahami keluhan/pertanyaan informal.
        """
        return (
            f"Pertanyaan pengguna:\n{question}\n\n"
            "Cari secara semantik: pahami maksud, bukan sekadar kata yang sama persis.\n"
            "Untuk pertanyaan definisi, cari di bagian pengantar, tujuan, atau gambaran umum.\n"
            "Untuk keluhan atau masalah, reformulasikan menjadi tujuan yang ingin dicapai "
            "(misal: cara login, syarat akses, langkah validasi) lalu cari rujukannya.\n"
            "Untuk prosedur, prioritaskan rujukan yang menyebut langkah, menu, form, tombol, status, field, atau syarat yang benar-benar relevan.\n"
            "Jawab langsung dan fokus pada pertanyaan pengguna.\n"
            "Tetap pada objek yang ditanyakan. Jangan melebar ke data atau proses lain yang tidak diminta hanya karena masih berdekatan konteksnya.\n"
            "Jika dokumen menyebut alternatif atau data lain, masukkan hanya bila memang diperlukan agar jawaban inti menjadi benar.\n"
            "Jika rujukan mendukung, beri jawaban yang jelas, padat, dan relevan.\n"
            "Jangan berhenti pada jawaban umum jika dokumen memuat langkah, syarat, pengecekan, status, lokasi menu, nama tombol, atau isian form yang lebih rinci.\n"
            "Utamakan langkah, menu, tombol, status, syarat, hak akses, dan isian form yang paling membantu menjawab pertanyaan.\n"
            "Jika tidak ada rujukan yang jelas, jawab bahwa jawaban belum tersedia di dokumen."
        )

    @staticmethod
    def _build_recovery_prompt(question: str) -> str:
        """
        Prompt cadangan — dipakai saat pass pertama gagal menemukan grounding.
        Mendorong model untuk mereformulasi keluhan menjadi pencarian yang lebih eksplisit.
        """
        return (
            f"Pertanyaan atau keluhan pengguna:\n{question}\n\n"
            "Jika kalimat di atas berupa keluhan atau laporan masalah, ubah menjadi pertanyaan eksplisit "
            "tentang cara, langkah, syarat, hak akses, status, atau penyebab yang relevan.\n"
            "Gunakan hanya rujukan yang benar-benar ditemukan dari File Search dokumen KRISNA.\n"
            "Jawab langsung, fokus pada masalah yang ditanyakan, secara jelas, padat, dan relevan.\n"
            "Jangan mengasumsikan maksud lain dan jangan melebar ke proses atau data lain yang tidak diminta jika tidak diperlukan.\n"
            "Jika dokumen menjelaskan prosesnya, tulis langkah inti secara berurutan, cukup detail, dan tanpa bertele-tele.\n"
            "Jika ada langkah penyelesaian, sebutkan syarat, status, lokasi menu, nama tombol, atau batasan penting yang benar-benar relevan.\n"
            "Jika tidak ada rujukan yang jelas, jawab bahwa jawaban belum tersedia di dokumen."
        )

    # ------------------------------------------------------------------
    # Core API call
    # ------------------------------------------------------------------

    def _call_model(self, prompt: str, model_name: str, store_names: list[str]):
        return self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0,
                max_output_tokens=self._max_tokens,
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

    def _call_model_with_retry(self, prompt: str, model_name: str, store_names: list[str]):
        attempts = self._retry_attempts + 1
        last_exc: Exception | None = None

        for attempt in range(attempts):
            try:
                return self._call_model(prompt, model_name, store_names)
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable(exc) or attempt >= attempts - 1:
                    raise

                delay = self._retry_backoff_seconds * (attempt + 1)
                logger.warning(
                    "[RAG] Retryable model error, retry %d/%d dalam %.1fs: %s",
                    attempt + 1,
                    self._retry_attempts,
                    delay,
                    exc,
                )
                if delay > 0:
                    time.sleep(delay)

        if last_exc is not None:
            raise last_exc

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response) -> dict[str, Any]:
        answer = self._clean_answer(str(getattr(response, "text", "") or ""))

        grounding_chunks = self.file_search_service.extract_grounding_chunks(response)
        used_files = self.file_search_service.extract_used_files(response)
        has_grounding = self.file_search_service.has_grounding(response)

        logger.debug("[RAG] parse | len=%d grounding=%s files=%s", len(answer), has_grounding, used_files)

        if not has_grounding:
            # Model merespons tapi tidak ada dokumen yang dirujuk —
            # tolak jawaban agar tidak muncul halusinasi.
            if answer:
                logger.warning(
                    "[RAG] Menolak jawaban tanpa grounding terkonfirmasi (len=%d).",
                    len(answer),
                )
                return self._fail(
                    "Saya belum bisa memastikan rujukan jawaban dari dokumen, jadi jawaban tidak saya tampilkan.",
                    "missing_grounding_metadata",
                )
            return self._fail(
                "Saya belum menemukan rujukan yang cukup relevan pada dokumen yang tersedia.",
                "context_not_found",
            )

        if not answer:
            return self._fail(
                "Rujukan ditemukan, tapi saya belum bisa merangkai jawaban yang jelas. "
                "Coba pertanyaan yang lebih spesifik.",
                "empty_model_text",
                used_files,
                grounding_chunks,
            )

        return self._ok(answer, used_files, grounding_chunks)

    # ------------------------------------------------------------------
    # Generate answer
    # ------------------------------------------------------------------

    def _generate_answer(self, prompt: str) -> dict[str, Any]:
        model_name = self._model_name()
        if not model_name:
            return self._fail("MODEL_NAME belum diisi di konfigurasi.", "model_not_found")

        store_names = self._get_store_names()
        if not store_names:
            return self._fail(
                "Dokumen KRISNA belum siap diakses. Pastikan file sudah di-upload dan di-ingest.",
                "store_not_ready",
            )

        try:
            response = self._call_model_with_retry(prompt, model_name, store_names)
        except ServerError as exc:
            logger.warning("[RAG] ServerError: %s", exc)
            if self._is_retryable(exc):
                return self._resp_provider_unavailable()
            return self._fail("Layanan Gemini sedang bermasalah. Coba lagi beberapa saat.", "server_error")
        except ClientError as exc:
            msg = str(exc)
            logger.error("[RAG] ClientError: %s", msg)
            if model_name.casefold() in msg.casefold() and "not found" in msg.casefold():
                return self._fail(
                    f"Model '{model_name}' tidak tersedia. Periksa MODEL_NAME di konfigurasi.",
                    "model_not_found",
                )
            if self._is_retryable(exc):
                return self._resp_provider_unavailable()
            return self._fail("Terjadi kesalahan saat memproses jawaban. Coba lagi beberapa saat.", "client_error")
        except Exception as exc:
            if self._is_retryable(exc):
                logger.warning("[RAG] Retryable provider error: %s", exc)
                return self._resp_provider_unavailable()
            logger.exception("[RAG] Unexpected error: %s", exc)
            return self._fail(
                "Terjadi kesalahan tak terduga saat memproses jawaban. Coba lagi beberapa saat.",
                "unexpected_generation_error",
            )

        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_chat(self, question: str, answer: str, found: bool, used_files: list[str], error: str = "") -> None:
        try:
            self.chat_log_repository.create(question=question, answer=answer, found=found, used_files=used_files)
        except Exception as exc:
            logger.warning("[RAG] DB log gagal: %s", exc)
        try:
            self.audit_logger.write({"question": question, "found": found, "used_files": used_files, "error": error})
        except Exception as exc:
            logger.warning("[RAG] Audit log gagal: %s", exc)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def ask(self, question: str) -> dict[str, Any]:
        question = (question or "").strip()

        if not question:
            result = self._fail("Pertanyaan tidak boleh kosong.", "empty_question")
            self._log_chat(question, result["message"], False, [], result["error"])
            return result

        # Sapaan / penutup
        social = self._detect_social(question)
        if social:
            self._log_chat(question, social["answer"], True, [], "")
            return social

        # Dokumen belum siap
        if not self._get_indexed_files():
            result = self._fail(
                self._helpdesk_message(),
                "context_not_found",
            )
            self._log_chat(question, result["message"], False, [], result["error"])
            return result

        # Pass 1 — prompt utama
        result = self._generate_answer(self._build_prompt(question))

        # Pass 2 — recovery prompt jika dokumen ada tapi grounding tidak terdeteksi
        # (membantu pertanyaan informal / keluhan yang kata-katanya tidak persis sama dengan dokumen)
        if not result.get("found") and result.get("error") in {"context_not_found", "missing_grounding_metadata"}:
            recovery = self._generate_answer(self._build_recovery_prompt(question))
            if recovery.get("found"):
                result = recovery

        if result.get("found") and self._looks_like_unanswered_answer(str(result.get("answer", "") or "")):
            result = self._fail(self._helpdesk_message(), "context_not_found")
        elif not result.get("found") and result.get("error") in {
            "context_not_found",
            "missing_grounding_metadata",
            "empty_model_text",
            "store_not_ready",
        }:
            result = self._fail(self._helpdesk_message(), str(result.get("error", "") or "context_not_found"))

        found      = bool(result.get("found") and result.get("answer"))
        answer     = str(result.get("answer", "") or "").strip() if found else ""
        message    = str(result.get("message", "") or "").strip() if not found else ""
        error      = str(result.get("error",   "") or "").strip()
        used_files = list(result.get("used_files") or []) if found else []
        grounding_chunks = list(result.get("grounding_chunks") or []) if found else []

        final = {
            "found":      found,
            "answer":     answer,
            "message":    message or ("" if found else "Saya belum menemukan jawaban yang sesuai di dokumen."),
            "error":      error,
            "used_files": used_files,
            "grounding_chunks": grounding_chunks,
        }
        self._log_chat(question, final["answer"] or final["message"], final["found"], final["used_files"], final["error"])
        return final
