from __future__ import annotations

from sqlalchemy import select

from app.db.models import IngestedDocumentRecord
from app.db.session import SessionLocal


class IngestedDocumentRepository:
    def list_all(self) -> list[IngestedDocumentRecord]:
        with SessionLocal() as session:
            return list(session.scalars(select(IngestedDocumentRecord).order_by(IngestedDocumentRecord.source_file)))

    def get(self, source_file: str) -> IngestedDocumentRecord | None:
        with SessionLocal() as session:
            return session.scalar(
                select(IngestedDocumentRecord).where(IngestedDocumentRecord.source_file == source_file)
            )

    def list_indexed_files(self) -> list[str]:
        with SessionLocal() as session:
            rows = session.scalars(
                select(IngestedDocumentRecord.source_file)
                .where(IngestedDocumentRecord.chunk_count > 0)
                .order_by(IngestedDocumentRecord.source_file)
            )
            return list(rows)

    def upsert(
        self,
        source_file: str,
        sha256: str,
        chunk_count: int,
        indexed_at: datetime,
        *,
        file_search_store_name: str | None = None,
        remote_document_name: str | None = None,
    ) -> None:
        with SessionLocal() as session:
            record = session.scalar(
                select(IngestedDocumentRecord).where(IngestedDocumentRecord.source_file == source_file)
            )
            if record is None:
                record = IngestedDocumentRecord(source_file=source_file)
                session.add(record)

            record.sha256 = sha256
            record.chunk_count = chunk_count
            record.indexed_at = indexed_at
            record.file_search_store_name = file_search_store_name
            record.remote_document_name = remote_document_name
            session.commit()

    def clear(self, source_file: str) -> None:
        with SessionLocal() as session:
            record = session.scalar(
                select(IngestedDocumentRecord).where(IngestedDocumentRecord.source_file == source_file)
            )
            if record is not None:
                session.delete(record)
                session.commit()
