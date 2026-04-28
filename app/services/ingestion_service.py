from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

from app.repositories.audit_repository import AuditRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.ingest_checkpoint_repository import IngestCheckpointRepository
from app.repositories.ingested_document_repository import IngestedDocumentRepository
from app.services.gemini_file_search_service import GeminiFileSearchService


class IngestionService:
    def __init__(self) -> None:
        self.document_repository = DocumentRepository()
        self.ingested_document_repository = IngestedDocumentRepository()
        self.ingest_checkpoint_repository = IngestCheckpointRepository()
        self.audit_repository = AuditRepository()
        self._file_search_service: GeminiFileSearchService | None = None

    def run_startup_cleanup(self) -> None:
        self._cleanup_orphaned_index_entries()
        self._cleanup_orphaned_checkpoints()

    def _cleanup_orphaned_index_entries(self) -> None:
        existing_names = set(self.document_repository.list_source_files())
        for item in self.ingested_document_repository.list_all():
            if item.source_file not in existing_names:
                self.ingested_document_repository.clear(item.source_file)

    def _cleanup_orphaned_checkpoints(self) -> None:
        existing_names = set(self.document_repository.list_source_files())
        for item in self.ingest_checkpoint_repository.list_all():
            if item.source_file not in existing_names:
                self.ingest_checkpoint_repository.clear(item.source_file)

    @property
    def file_search_service(self) -> GeminiFileSearchService:
        if self._file_search_service is None:
            self._file_search_service = GeminiFileSearchService()
        return self._file_search_service

    def _read_index(self) -> list[dict[str, Any]]:
        records = self.ingested_document_repository.list_all()
        return [
            {
                "source_file": item.source_file,
                "sha256": item.sha256,
                "chunk_count": item.chunk_count,
                "file_search_store_name": item.file_search_store_name,
                "remote_document_name": item.remote_document_name,
                "indexed_at": item.indexed_at.isoformat() if item.indexed_at else None,
            }
            for item in records
            if item.chunk_count > 0
        ]

    def _replace_index_entry(self, source_file: str, new_entry: dict[str, Any]) -> None:
        self.ingested_document_repository.upsert(
            source_file=source_file,
            sha256=str(new_entry["sha256"]),
            chunk_count=int(new_entry["chunk_count"]),
            file_search_store_name=str(new_entry.get("file_search_store_name", "") or "") or None,
            remote_document_name=str(new_entry.get("remote_document_name", "") or "") or None,
            indexed_at=datetime.fromisoformat(str(new_entry["indexed_at"]).replace("Z", "+00:00")),
        )

    @staticmethod
    def _sha256_bytes(content: bytes) -> str:
        h = hashlib.sha256()
        for offset in range(0, len(content), 8192):
            h.update(content[offset : offset + 8192])
        return h.hexdigest()

    @staticmethod
    def _safe_name(filename: str) -> str:
        return Path(filename).name.replace("\\", "_").replace("/", "_")

    def _unique_name(self, filename: str) -> str:
        candidate = self._safe_name(filename)
        existing_names = set(self.document_repository.list_source_files())
        if candidate not in existing_names:
            return candidate

        safe_path = Path(candidate)
        stem = safe_path.stem
        suffix = safe_path.suffix
        counter = 1
        while True:
            next_candidate = f"{stem}-{counter}{suffix}"
            if next_candidate not in existing_names:
                return next_candidate
            counter += 1

    def _index_pdf(
        self,
        source_file: str,
        content: bytes,
        *,
        force: bool = False,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        del force

        file_hash = self._sha256_bytes(content)
        total_chunks = 1
        completed_chunks = 0

        self.ingest_checkpoint_repository.upsert(
            source_file=source_file,
            sha256=file_hash,
            total_chunks=total_chunks,
            completed_chunks=completed_chunks,
            status="running",
            error_message=None,
        )

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "chunk_progress",
                    "file": source_file,
                    "current_chunk": completed_chunks,
                    "current_file_total_chunks": total_chunks,
                    "resumed_from_chunk": 0,
                }
            )

        try:
            existing_record = self.ingested_document_repository.get(source_file)
            upload_result = self.file_search_service.upload_document(
                source_file=source_file,
                content=content,
                mime_type="application/pdf",
                sha256=file_hash,
            )
            if (
                existing_record is not None
                and existing_record.remote_document_name
                and existing_record.remote_document_name != upload_result["document_name"]
            ):
                try:
                    self.file_search_service.delete_document(existing_record.remote_document_name)
                except Exception:
                    # Remote metadata can outlive the API key/project that created it.
                    # Keep re-ingest successful after the new document is ACTIVE.
                    pass

            completed_chunks = total_chunks
            self.ingest_checkpoint_repository.upsert(
                source_file=source_file,
                sha256=file_hash,
                total_chunks=total_chunks,
                completed_chunks=completed_chunks,
                status="running",
                error_message=None,
            )

            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "chunk_progress",
                        "file": source_file,
                        "current_chunk": completed_chunks,
                        "current_file_total_chunks": total_chunks,
                        "resumed_from_chunk": 0,
                    }
                )
        except Exception as exc:
            self.ingest_checkpoint_repository.upsert(
                source_file=source_file,
                sha256=file_hash,
                total_chunks=total_chunks,
                completed_chunks=completed_chunks,
                status="failed",
                error_message=str(exc),
            )
            raise

        entry = {
            "source_file": source_file,
            "sha256": file_hash,
            "chunk_count": total_chunks,
            "file_search_store_name": upload_result["store_name"],
            "remote_document_name": upload_result["document_name"],
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._replace_index_entry(source_file, entry)
        self.ingest_checkpoint_repository.clear(source_file)
        return entry

    def save_files(self, files: Iterable[tuple[str, bytes]]) -> list[str]:
        saved_files: list[str] = []
        for original_name, content in files:
            target_name = self._unique_name(original_name)
            saved_files.append(target_name)
            self.document_repository.ensure_uploaded(
                target_name,
                mime_type="application/pdf",
                content=content,
            )
            self.audit_repository.create(
                "document_uploaded",
                {
                    "source_file": target_name,
                    "original_name": original_name,
                    "size_bytes": len(content),
                },
            )
        return saved_files

    def replace_file(self, source_file: str, filename: str, content: bytes) -> str:
        existing_content = self.document_repository.get_file_content(source_file)
        if existing_content is None:
            raise FileNotFoundError(f"File '{source_file}' tidak ditemukan.")

        safe_filename = self._safe_name(filename)
        target_name = source_file if safe_filename == source_file else safe_filename
        existing_names = set(self.document_repository.list_source_files())
        if target_name != source_file and target_name in existing_names:
            raise FileExistsError(f"File tujuan '{target_name}' sudah ada.")

        self._delete_remote_index(source_file)
        self.ingested_document_repository.clear(source_file)
        self.ingest_checkpoint_repository.clear(source_file)
        self.document_repository.replace_file(
            source_file,
            target_name=target_name,
            mime_type="application/pdf",
            content=content,
        )
        self.audit_repository.create(
            "document_replaced",
            {
                "source_file": source_file,
                "target_name": target_name,
                "size_bytes": len(content),
            },
        )
        return target_name

    def _delete_remote_index(self, source_file: str) -> None:
        try:
            record = self.ingested_document_repository.get(source_file)
            if record is None:
                self.file_search_service.delete_document_by_source_file(source_file)
                return

            if record.remote_document_name:
                self.file_search_service.delete_document(record.remote_document_name)
                return

            self.file_search_service.delete_document_by_source_file(
                source_file,
                store_name=record.file_search_store_name,
            )
        except Exception:
            pass

    def delete_index(self, source_file: str) -> None:
        file_exists = self.document_repository.get_by_source_file(source_file) is not None
        index_exists = source_file in self.ingested_document_repository.list_indexed_files()
        checkpoint_exists = self.ingest_checkpoint_repository.get(source_file) is not None
        if not file_exists and not index_exists and not checkpoint_exists:
            raise FileNotFoundError(f"File atau index '{source_file}' tidak ditemukan.")

        self._delete_remote_index(source_file)
        self.ingested_document_repository.clear(source_file)
        self.ingest_checkpoint_repository.clear(source_file)
        self.audit_repository.create(
            "document_index_deleted",
            {
                "source_file": source_file,
            },
        )

    def delete_file(self, source_file: str) -> None:
        if self.document_repository.get_file_content(source_file) is None:
            raise FileNotFoundError(f"File '{source_file}' tidak ditemukan.")
        self._delete_remote_index(source_file)
        self.ingested_document_repository.clear(source_file)
        self.ingest_checkpoint_repository.clear(source_file)
        self.document_repository.delete(source_file)
        self.audit_repository.create(
            "document_deleted",
            {
                "source_file": source_file,
            },
        )

    def list_docs(self) -> list[str]:
        return self.document_repository.list_source_files()

    def list_indexed_files(self) -> list[str]:
        return self.ingested_document_repository.list_indexed_files()

    def list_file_search_store_names(self) -> list[str]:
        store_names: list[str] = []
        for item in self.ingested_document_repository.list_all():
            store_name = str(item.file_search_store_name or "").strip()
            if store_name and store_name not in store_names:
                store_names.append(store_name)
        return store_names

    def upload_all(
        self,
        force: bool = False,
        source_files: list[str] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        current = self._read_index()
        current_by_file = {item["source_file"]: item for item in current}
        all_pdfs = {
            record.source_file: record
            for record in self.document_repository.list_stored_documents()
        }

        if source_files is None:
            pdfs = sorted(all_pdfs.values(), key=lambda item: item.source_file)
        else:
            missing_files = [name for name in source_files if name not in all_pdfs]
            if missing_files:
                raise FileNotFoundError(
                    f"File tidak ditemukan untuk ingest: {', '.join(missing_files)}"
                )
            pdfs = [all_pdfs[name] for name in source_files]

        uploaded = 0
        skipped = 0
        failed_files: list[dict[str, str]] = []
        total_files = len(pdfs)

        for index, pdf in enumerate(pdfs, start=1):
            if not pdf.file_data:
                failed_files.append({"file": pdf.source_file, "error": "File PDF kosong di database."})
                continue

            file_hash = self._sha256_bytes(bytes(pdf.file_data))
            existing = current_by_file.get(pdf.source_file)
            has_remote_index = bool(
                existing
                and existing.get("chunk_count")
                and existing.get("remote_document_name")
            )

            if existing and existing.get("sha256") == file_hash and has_remote_index and not force:
                self.ingest_checkpoint_repository.clear(pdf.source_file)
                skipped += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "skipped",
                            "file": pdf.source_file,
                            "processed_files": index,
                            "total_files": total_files,
                            "uploaded": uploaded,
                            "skipped": skipped,
                            "failed": len(failed_files),
                            "current_chunk": 0,
                            "current_file_total_chunks": 0,
                            "resumed_from_chunk": 0,
                        }
                    )
                continue

            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "processing",
                        "file": pdf.source_file,
                        "processed_files": index - 1,
                        "total_files": total_files,
                        "uploaded": uploaded,
                        "skipped": skipped,
                        "failed": len(failed_files),
                        "current_chunk": 0,
                        "current_file_total_chunks": 0,
                        "resumed_from_chunk": 0,
                    }
                )

            try:
                file_progress_callback: Callable[[dict[str, Any]], None] | None = None
                if progress_callback is not None:
                    def file_progress_callback(event: dict[str, Any]) -> None:
                        progress_callback(
                            {
                                "processed_files": index - 1,
                                "total_files": total_files,
                                "uploaded": uploaded,
                                "skipped": skipped,
                                "failed": len(failed_files),
                                **event,
                            }
                        )

                self._index_pdf(
                    pdf.source_file,
                    bytes(pdf.file_data),
                    force=force,
                    progress_callback=file_progress_callback,
                )
                uploaded += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "uploaded",
                            "file": pdf.source_file,
                            "processed_files": index,
                            "total_files": total_files,
                            "uploaded": uploaded,
                            "skipped": skipped,
                            "failed": len(failed_files),
                            "current_chunk": 0,
                            "current_file_total_chunks": 0,
                            "resumed_from_chunk": 0,
                        }
                    )
            except Exception as exc:
                failed_files.append({"file": pdf.source_file, "error": str(exc)})
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "failed",
                            "file": pdf.source_file,
                            "error": str(exc),
                            "processed_files": index,
                            "total_files": total_files,
                            "uploaded": uploaded,
                            "skipped": skipped,
                            "failed": len(failed_files),
                            "current_chunk": 0,
                            "current_file_total_chunks": 0,
                            "resumed_from_chunk": 0,
                        }
                    )

        indexed_files = len(self._read_index())
        return {
            "running": False,
            "status": "completed",
            "message": "Ingest selesai.",
            "force": force,
            "target_files": [pdf.source_file for pdf in pdfs],
            "uploaded": uploaded,
            "skipped": skipped,
            "failed": len(failed_files),
            "failed_files": failed_files,
            "indexed_files": indexed_files,
            "total_files": total_files,
            "processed_files": total_files,
            "current_file": "",
        }


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    return IngestionService()
