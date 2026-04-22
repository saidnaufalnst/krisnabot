from __future__ import annotations

import threading
from copy import deepcopy
from typing import Any

from app.services.ingestion_service import IngestionService


class IngestJobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = self._new_state()

    @staticmethod
    def _new_state() -> dict[str, Any]:
        return {
            "running": False,
            "status": "idle",
            "message": "Belum ada proses ingest.",
            "force": False,
            "target_files": [],
            "processed_files": 0,
            "total_files": 0,
            "current_file": "",
            "current_chunk": 0,
            "current_file_total_chunks": 0,
            "resumed_from_chunk": 0,
            "uploaded": 0,
            "skipped": 0,
            "failed": 0,
            "failed_files": [],
            "indexed_files": 0,
        }

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

    def start(self, force: bool = False, source_files: list[str] | None = None) -> dict[str, Any]:
        with self._lock:
            if self._state["running"]:
                return deepcopy(self._state)

            self._state = self._new_state()
            self._state.update(
                {
                    "running": True,
                    "status": "running",
                    "message": "Proses ingest sedang berjalan.",
                    "force": force,
                    "target_files": source_files or [],
                }
            )

        worker = threading.Thread(
            target=self._run_job,
            args=(force, source_files),
            daemon=True,
        )
        worker.start()
        return self.get_status()

    def _update_progress(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._state["processed_files"] = event.get("processed_files", self._state["processed_files"])
            self._state["total_files"] = event.get("total_files", self._state["total_files"])
            self._state["uploaded"] = event.get("uploaded", self._state["uploaded"])
            self._state["skipped"] = event.get("skipped", self._state["skipped"])
            self._state["failed"] = event.get("failed", self._state["failed"])
            self._state["current_file"] = event.get("file", self._state["current_file"])
            self._state["current_chunk"] = event.get("current_chunk", self._state["current_chunk"])
            self._state["current_file_total_chunks"] = event.get(
                "current_file_total_chunks",
                self._state["current_file_total_chunks"],
            )
            self._state["resumed_from_chunk"] = event.get(
                "resumed_from_chunk",
                self._state["resumed_from_chunk"],
            )

            event_name = event.get("event")
            file_name = event.get("file", "")

            if event_name == "processing":
                self._state["message"] = f"Sedang memproses {file_name}..."
            elif event_name == "chunk_progress":
                current_chunk = int(event.get("current_chunk", 0) or 0)
                total_chunks = int(event.get("current_file_total_chunks", 0) or 0)
                resumed_from_chunk = int(event.get("resumed_from_chunk", 0) or 0)
                if resumed_from_chunk > 0 and current_chunk <= resumed_from_chunk:
                    self._state["message"] = (
                        f"Melanjutkan ingest {file_name} dari chunk {resumed_from_chunk}/{total_chunks}..."
                    )
                else:
                    self._state["message"] = (
                        f"Memproses {file_name}: chunk {current_chunk}/{total_chunks}."
                    )
            elif event_name == "uploaded":
                self._state["message"] = f"Berhasil ingest {file_name}."
            elif event_name == "skipped":
                self._state["message"] = f"Melewati {file_name} karena tidak ada perubahan."
            elif event_name == "failed":
                self._state["message"] = f"Gagal ingest {file_name}: {event.get('error', 'unknown error')}"

    def _run_job(self, force: bool, source_files: list[str] | None) -> None:
        service = IngestionService()
        try:
            result = service.upload_all(
                force=force,
                source_files=source_files,
                progress_callback=self._update_progress,
            )
            with self._lock:
                self._state.update(result)
                self._state["running"] = False
                self._state["status"] = "completed"
                self._state["current_file"] = ""
                self._state["current_chunk"] = 0
                self._state["current_file_total_chunks"] = 0
                self._state["resumed_from_chunk"] = 0
                self._state["message"] = (
                    f"Ingest selesai untuk {len(source_files)} file."
                    if source_files
                    else "Ingest selesai."
                )
        except Exception as exc:
            with self._lock:
                self._state["running"] = False
                self._state["status"] = "failed"
                self._state["current_file"] = ""
                self._state["current_chunk"] = 0
                self._state["current_file_total_chunks"] = 0
                self._state["resumed_from_chunk"] = 0
                self._state["message"] = f"Ingest gagal: {exc}"


ingest_job_manager = IngestJobManager()
