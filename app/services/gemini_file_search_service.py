from __future__ import annotations

import io
import time
from typing import Any

from google.genai import types
from google.genai.errors import ClientError

from app.core.config import settings
from app.services.gemini_client import get_gemini_client


_cached_store_name: str | None = None


class GeminiFileSearchService:
    def __init__(self) -> None:
        self.client = get_gemini_client(kind="chat")

    @staticmethod
    def _operation_error_message(operation: Any) -> str:
        error = getattr(operation, "error", None)
        if error is None:
            return "Unknown File Search operation error."

        message = getattr(error, "message", None)
        if message:
            return str(message)
        return str(error)

    @staticmethod
    def _field_value(source: Any, *names: str) -> Any:
        if source is None:
            return None
        if isinstance(source, dict):
            for name in names:
                if name in source:
                    return source[name]
            return None
        for name in names:
            value = getattr(source, name, None)
            if value is not None:
                return value
        return None

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return []

    @staticmethod
    def _custom_metadata_value(metadata: Any, key: str) -> str | None:
        if not metadata or GeminiFileSearchService._field_value(metadata, "key") != key:
            return None

        string_value = GeminiFileSearchService._field_value(metadata, "string_value", "stringValue")
        if string_value is not None:
            return str(string_value)

        numeric_value = GeminiFileSearchService._field_value(metadata, "numeric_value", "numericValue")
        if numeric_value is not None:
            return str(numeric_value)

        string_list = GeminiFileSearchService._field_value(
            metadata,
            "string_list_value",
            "stringListValue",
        )
        values = GeminiFileSearchService._field_value(string_list, "values") or []
        values = GeminiFileSearchService._as_list(values)
        if values:
            return str(values[0])

        return None

    @staticmethod
    def _source_file_from_document(document: Any) -> str | None:
        custom_metadata = GeminiFileSearchService._field_value(
            document,
            "custom_metadata",
            "customMetadata",
        ) or []
        for item in custom_metadata:
            value = GeminiFileSearchService._custom_metadata_value(item, "source_file")
            if value:
                return value

        display_name = GeminiFileSearchService._field_value(document, "display_name", "displayName")
        if display_name:
            return str(display_name)

        return None

    @staticmethod
    def _source_file_from_chunk(chunk: Any) -> str | None:
        retrieved_context = GeminiFileSearchService._field_value(
            chunk,
            "retrieved_context",
            "retrievedContext",
        )
        if retrieved_context is None:
            return None

        custom_metadata = GeminiFileSearchService._field_value(
            retrieved_context,
            "custom_metadata",
            "customMetadata",
        ) or []
        for item in custom_metadata:
            value = GeminiFileSearchService._custom_metadata_value(item, "source_file")
            if value:
                return value

        title = GeminiFileSearchService._field_value(retrieved_context, "title")
        if title:
            return str(title)

        uri = GeminiFileSearchService._field_value(retrieved_context, "uri")
        if uri:
            return str(uri)

        return None

    @staticmethod
    def _text_from_chunk(chunk: Any) -> str:
        retrieved_context = GeminiFileSearchService._field_value(
            chunk,
            "retrieved_context",
            "retrievedContext",
        )
        if retrieved_context is None:
            return ""

        text = GeminiFileSearchService._field_value(retrieved_context, "text")
        if text:
            return str(text)
        return ""

    @staticmethod
    def _grounding_chunks_from_response(response: Any) -> list[Any]:
        candidates = GeminiFileSearchService._field_value(response, "candidates") or []
        chunks: list[Any] = []
        for candidate in candidates:
            grounding_metadata = GeminiFileSearchService._field_value(
                candidate,
                "grounding_metadata",
                "groundingMetadata",
            )
            grounding_chunks = GeminiFileSearchService._field_value(
                grounding_metadata,
                "grounding_chunks",
                "groundingChunks",
            ) or []
            chunks.extend(GeminiFileSearchService._as_list(grounding_chunks))
        return chunks

    @staticmethod
    def _document_state(document: Any) -> str:
        state = GeminiFileSearchService._field_value(document, "state")
        if state is None:
            return ""
        enum_name = getattr(state, "name", None)
        if enum_name:
            return str(enum_name).strip().upper()

        normalized = str(state).strip()
        if "." in normalized:
            normalized = normalized.rsplit(".", 1)[-1]
        return normalized.upper()

    @staticmethod
    def _document_name_from_operation_response(response: Any) -> str:
        document_name = GeminiFileSearchService._field_value(
            response,
            "document_name",
            "documentName",
            "name",
        )
        if document_name:
            return str(document_name).strip()
        return ""

    @staticmethod
    def _chunking_config() -> types.ChunkingConfig | None:
        max_tokens = int(getattr(settings, "file_search_max_tokens_per_chunk", 300) or 0)
        if max_tokens <= 0:
            return None

        overlap = int(getattr(settings, "file_search_max_overlap_tokens", 40) or 0)
        overlap = max(0, min(overlap, max_tokens - 1))
        return types.ChunkingConfig(
            white_space_config=types.WhiteSpaceConfig(
                max_tokens_per_chunk=max_tokens,
                max_overlap_tokens=overlap,
            )
        )

    def _wait_for_operation(self, operation: Any) -> Any:
        poll_interval = max(float(getattr(settings, "file_search_poll_interval_seconds", 2) or 2), 0.2)
        timeout_seconds = max(float(getattr(settings, "file_search_operation_timeout_seconds", 300) or 300), 5.0)
        deadline = time.monotonic() + timeout_seconds
        current_operation = operation

        while not getattr(current_operation, "done", False):
            if time.monotonic() >= deadline:
                raise TimeoutError("Operasi Gemini File Search melebihi batas waktu.")
            time.sleep(poll_interval)
            current_operation = self.client.operations.get(current_operation)

        if getattr(current_operation, "error", None):
            raise RuntimeError(self._operation_error_message(current_operation))

        return current_operation

    def _wait_for_document_active(self, document_name: str) -> Any:
        if not document_name:
            raise ValueError("Document name tidak boleh kosong.")

        poll_interval = max(
            float(getattr(settings, "file_search_document_poll_interval_seconds", 2) or 2),
            0.2,
        )
        timeout_seconds = max(
            float(getattr(settings, "file_search_document_ready_timeout_seconds", 300) or 300),
            5.0,
        )
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None

        while True:
            try:
                document = self.client.file_search_stores.documents.get(name=document_name)
            except Exception as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Dokumen Gemini '{document_name}' belum bisa diakses sampai batas waktu."
                    ) from exc
                time.sleep(poll_interval)
                continue

            state = self._document_state(document)
            if state in {"STATE_ACTIVE", "ACTIVE"}:
                return document
            if state in {"STATE_FAILED", "FAILED"}:
                raise RuntimeError(f"Dokumen Gemini '{document_name}' gagal diproses oleh File Search.")
            if time.monotonic() >= deadline:
                detail = f" State terakhir: {state}." if state else ""
                raise TimeoutError(
                    f"Dokumen Gemini '{document_name}' belum ACTIVE sampai batas waktu.{detail}"
                ) from last_error
            time.sleep(poll_interval)

    def get_or_create_store_name(self) -> str:
        global _cached_store_name

        configured_store = str(getattr(settings, "file_search_store", "") or "").strip()
        if configured_store.startswith("fileSearchStores/"):
            _cached_store_name = configured_store
            return configured_store

        if _cached_store_name:
            return _cached_store_name

        display_name = configured_store or "krisnabot-store"
        if not display_name:
            raise RuntimeError("FILE_SEARCH_STORE belum diisi.")

        try:
            for store in self.client.file_search_stores.list():
                if str(getattr(store, "display_name", "") or "").strip() == display_name:
                    store_name = str(getattr(store, "name", "") or "").strip()
                    if store_name:
                        _cached_store_name = store_name
                        return store_name
        except Exception:
            pass

        store = self.client.file_search_stores.create(
            config=types.CreateFileSearchStoreConfig(display_name=display_name)
        )
        store_name = str(getattr(store, "name", "") or "").strip()
        if not store_name:
            raise RuntimeError("Gemini File Search store gagal dibuat.")

        _cached_store_name = store_name
        return store_name

    def find_document_name_by_source_file(self, source_file: str, *, store_name: str | None = None) -> str | None:
        parent = store_name or self.get_or_create_store_name()
        try:
            for document in self.client.file_search_stores.documents.list(parent=parent):
                if self._source_file_from_document(document) == source_file:
                    document_name = str(getattr(document, "name", "") or "").strip()
                    if document_name:
                        return document_name
        except Exception:
            return None
        return None

    def upload_document(
        self,
        *,
        source_file: str,
        content: bytes,
        mime_type: str,
        sha256: str | None = None,
    ) -> dict[str, str]:
        if not content:
            raise ValueError(f"Konten file '{source_file}' kosong.")

        store_name = self.get_or_create_store_name()
        buffer = io.BytesIO(content)
        buffer.name = source_file

        custom_metadata = [
            types.CustomMetadata(key="source_file", string_value=source_file),
        ]
        if sha256:
            custom_metadata.append(types.CustomMetadata(key="sha256", string_value=sha256))

        config_kwargs: dict[str, Any] = {
            "mime_type": mime_type,
            "display_name": source_file,
            "custom_metadata": custom_metadata,
        }
        chunking_config = self._chunking_config()
        if chunking_config is not None:
            config_kwargs["chunking_config"] = chunking_config

        operation = self.client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store_name,
            file=buffer,
            config=types.UploadToFileSearchStoreConfig(**config_kwargs),
        )
        completed_operation = self._wait_for_operation(operation)
        response = getattr(completed_operation, "response", None)
        document_name = self._document_name_from_operation_response(response)
        if not document_name:
            document_name = self.find_document_name_by_source_file(source_file, store_name=store_name) or ""
        if not document_name:
            raise RuntimeError(f"Dokumen Gemini untuk '{source_file}' tidak ditemukan setelah upload.")
        self._wait_for_document_active(document_name)

        return {
            "store_name": store_name,
            "document_name": document_name,
        }

    def delete_document(self, document_name: str) -> None:
        if not document_name:
            return

        try:
            self.client.file_search_stores.documents.delete(
                name=document_name,
                config=types.DeleteDocumentConfig(force=True),
            )
        except ClientError as exc:
            if "not found" in str(exc).lower():
                return
            raise

    def delete_document_by_source_file(self, source_file: str, *, store_name: str | None = None) -> None:
        document_name = self.find_document_name_by_source_file(source_file, store_name=store_name)
        if document_name:
            self.delete_document(document_name)

    def extract_used_files(self, response: Any) -> list[str]:
        used_files: list[str] = []

        for chunk in self._grounding_chunks_from_response(response):
            source_file = self._source_file_from_chunk(chunk)
            if source_file and source_file not in used_files:
                used_files.append(source_file)

        return used_files

    def extract_grounding_chunks(self, response: Any) -> list[dict[str, str]]:
        grounding_chunks: list[dict[str, str]] = []

        for chunk in self._grounding_chunks_from_response(response):
            source_file = self._source_file_from_chunk(chunk) or ""
            text = self._text_from_chunk(chunk)
            if not source_file and not text:
                continue
            item = {
                "source_file": source_file,
                "text": text,
            }
            if item not in grounding_chunks:
                grounding_chunks.append(item)

        return grounding_chunks

    def has_grounding(self, response: Any) -> bool:
        return bool(self._grounding_chunks_from_response(response))
