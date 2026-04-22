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
    def _custom_metadata_value(metadata: Any, key: str) -> str | None:
        if not metadata or getattr(metadata, "key", None) != key:
            return None

        string_value = getattr(metadata, "string_value", None)
        if string_value is not None:
            return str(string_value)

        numeric_value = getattr(metadata, "numeric_value", None)
        if numeric_value is not None:
            return str(numeric_value)

        values = getattr(metadata, "string_list_value", None) or []
        if values:
            return str(values[0])

        return None

    @staticmethod
    def _source_file_from_document(document: Any) -> str | None:
        custom_metadata = getattr(document, "custom_metadata", None) or []
        for item in custom_metadata:
            value = GeminiFileSearchService._custom_metadata_value(item, "source_file")
            if value:
                return value

        display_name = getattr(document, "display_name", None)
        if display_name:
            return str(display_name)

        return None

    @staticmethod
    def _source_file_from_chunk(chunk: Any) -> str | None:
        retrieved_context = getattr(chunk, "retrieved_context", None)
        if retrieved_context is None:
            return None

        custom_metadata = getattr(retrieved_context, "custom_metadata", None) or []
        for item in custom_metadata:
            value = GeminiFileSearchService._custom_metadata_value(item, "source_file")
            if value:
                return value

        title = getattr(retrieved_context, "title", None)
        if title:
            return str(title)

        uri = getattr(retrieved_context, "uri", None)
        if uri:
            return str(uri)

        return None

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

    def get_or_create_store_name(self) -> str:
        global _cached_store_name

        configured_name = str(getattr(settings, "file_search_store_name", "") or "").strip()
        if configured_name:
            _cached_store_name = configured_name
            return configured_name

        if _cached_store_name:
            return _cached_store_name

        display_name = str(getattr(settings, "file_search_store_display_name", "") or "").strip()
        if not display_name:
            raise RuntimeError("FILE_SEARCH_STORE_DISPLAY_NAME belum diisi.")

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

        operation = self.client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store_name,
            file=buffer,
            config=types.UploadToFileSearchStoreConfig(
                mime_type=mime_type,
                display_name=source_file,
                custom_metadata=custom_metadata,
            ),
        )
        completed_operation = self._wait_for_operation(operation)
        response = getattr(completed_operation, "response", None)
        document_name = str(getattr(response, "document_name", "") or "").strip()
        if not document_name:
            document_name = self.find_document_name_by_source_file(source_file, store_name=store_name) or ""
        if not document_name:
            raise RuntimeError(f"Dokumen Gemini untuk '{source_file}' tidak ditemukan setelah upload.")

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
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return []

        grounding_metadata = getattr(candidates[0], "grounding_metadata", None)
        chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
        used_files: list[str] = []

        for chunk in chunks:
            source_file = self._source_file_from_chunk(chunk)
            if source_file and source_file not in used_files:
                used_files.append(source_file)

        return used_files

    def has_grounding(self, response: Any) -> bool:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return False

        grounding_metadata = getattr(candidates[0], "grounding_metadata", None)
        chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
        return bool(chunks)
