from __future__ import annotations

from sqlalchemy import select

from app.db.models import IngestCheckpointRecord
from app.db.session import SessionLocal


class IngestCheckpointRepository:
    def list_all(self) -> list[IngestCheckpointRecord]:
        with SessionLocal() as session:
            return list(
                session.scalars(
                    select(IngestCheckpointRecord).order_by(IngestCheckpointRecord.source_file)
                )
            )

    def get(self, source_file: str) -> IngestCheckpointRecord | None:
        with SessionLocal() as session:
            return session.scalar(
                select(IngestCheckpointRecord).where(IngestCheckpointRecord.source_file == source_file)
            )

    def upsert(
        self,
        *,
        source_file: str,
        sha256: str | None,
        total_chunks: int,
        completed_chunks: int,
        status: str,
        error_message: str | None = None,
    ) -> None:
        with SessionLocal() as session:
            record = session.scalar(
                select(IngestCheckpointRecord).where(IngestCheckpointRecord.source_file == source_file)
            )
            if record is None:
                record = IngestCheckpointRecord(source_file=source_file)
                session.add(record)

            record.sha256 = sha256
            record.total_chunks = int(total_chunks)
            record.completed_chunks = int(completed_chunks)
            record.status = str(status or "pending")
            record.error_message = error_message or None
            session.commit()

    def clear(self, source_file: str) -> None:
        with SessionLocal() as session:
            record = session.scalar(
                select(IngestCheckpointRecord).where(IngestCheckpointRecord.source_file == source_file)
            )
            if record is not None:
                session.delete(record)
                session.commit()
