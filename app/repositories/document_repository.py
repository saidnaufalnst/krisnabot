from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from app.db.models import UploadedDocumentRecord
from app.db.session import SessionLocal


class DocumentRepository:
    def list_all(self) -> list[UploadedDocumentRecord]:
        with SessionLocal() as session:
            return list(session.scalars(select(UploadedDocumentRecord).order_by(UploadedDocumentRecord.source_file)))

    def list_source_files(self) -> list[str]:
        with SessionLocal() as session:
            rows = session.scalars(select(UploadedDocumentRecord.source_file).order_by(UploadedDocumentRecord.source_file))
            return list(rows)

    def get_by_source_file(self, source_file: str) -> UploadedDocumentRecord | None:
        with SessionLocal() as session:
            return session.scalar(
                select(UploadedDocumentRecord).where(UploadedDocumentRecord.source_file == source_file)
            )

    def ensure_uploaded(
        self,
        source_file: str,
        *,
        mime_type: str | None = None,
        content: bytes | None = None,
    ) -> None:
        with SessionLocal() as session:
            existing = session.scalar(
                select(UploadedDocumentRecord).where(UploadedDocumentRecord.source_file == source_file)
            )
            if existing is None:
                file_data = content if content is not None else None
                session.add(
                    UploadedDocumentRecord(
                        source_file=source_file,
                        mime_type=mime_type or "application/pdf",
                        file_size=len(file_data or b""),
                        file_data=file_data,
                        uploaded_at=datetime.now(timezone.utc),
                    )
                )
            else:
                if mime_type is not None:
                    existing.mime_type = mime_type
                if content is not None:
                    existing.file_data = content
                    existing.file_size = len(content)
            session.commit()

    def replace_file(
        self,
        source_file: str,
        *,
        target_name: str,
        mime_type: str,
        content: bytes,
    ) -> None:
        with SessionLocal() as session:
            record = session.scalar(
                select(UploadedDocumentRecord).where(UploadedDocumentRecord.source_file == source_file)
            )
            if record is None:
                raise FileNotFoundError(f"File '{source_file}' tidak ditemukan.")

            record.source_file = target_name
            record.mime_type = mime_type
            record.file_size = len(content)
            record.file_data = content
            session.commit()

    def get_file_content(self, source_file: str) -> bytes | None:
        with SessionLocal() as session:
            record = session.scalar(
                select(UploadedDocumentRecord).where(UploadedDocumentRecord.source_file == source_file)
            )
            if record is None:
                return None
            return bytes(record.file_data or b"")

    def list_stored_documents(self) -> list[UploadedDocumentRecord]:
        with SessionLocal() as session:
            return list(
                session.scalars(
                    select(UploadedDocumentRecord)
                    .where(UploadedDocumentRecord.file_data.is_not(None))
                    .order_by(UploadedDocumentRecord.source_file)
                )
            )

    def delete(self, source_file: str) -> None:
        with SessionLocal() as session:
            record = session.scalar(
                select(UploadedDocumentRecord).where(UploadedDocumentRecord.source_file == source_file)
            )
            if record is not None:
                session.delete(record)
                session.commit()
