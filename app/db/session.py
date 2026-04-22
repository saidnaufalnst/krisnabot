from __future__ import annotations

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.base import Base


engine = create_engine(
    settings.database_url,
    echo=settings.database_echo,
    future=True,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def _migrate_documents_table() -> None:
    inspector = inspect(engine)
    if "documents" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("documents")}
    statements: list[str] = []

    if "mime_type" not in existing_columns:
        statements.append("ALTER TABLE documents ADD COLUMN mime_type VARCHAR(100)")
    if "file_size" not in existing_columns:
        statements.append("ALTER TABLE documents ADD COLUMN file_size INTEGER NOT NULL DEFAULT 0")
    if "file_data" not in existing_columns:
        statements.append("ALTER TABLE documents ADD COLUMN file_data BYTEA")

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def _migrate_ingested_documents_data() -> None:
    inspector = inspect(engine)
    if "documents" not in inspector.get_table_names() or "ingested_documents" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("documents")}
    required_columns = {"source_file", "sha256", "chunk_count", "indexed_at"}
    if not required_columns.issubset(existing_columns):
        return

    with engine.begin() as connection:
        existing_count = connection.execute(text("SELECT COUNT(*) FROM ingested_documents")).scalar_one()
        if existing_count:
            return

        connection.execute(
            text(
                """
                INSERT INTO ingested_documents (source_file, sha256, chunk_count, indexed_at, created_at, updated_at)
                SELECT source_file, sha256, chunk_count, indexed_at, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                FROM documents
                WHERE chunk_count > 0
                """
            )
        )


def _migrate_ingested_documents_table() -> None:
    inspector = inspect(engine)
    if "ingested_documents" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("ingested_documents")}
    statements: list[str] = []

    if "file_search_store_name" not in existing_columns:
        statements.append("ALTER TABLE ingested_documents ADD COLUMN file_search_store_name VARCHAR(255)")
    if "remote_document_name" not in existing_columns:
        statements.append("ALTER TABLE ingested_documents ADD COLUMN remote_document_name VARCHAR(255)")

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def _cleanup_documents_table() -> None:
    inspector = inspect(engine)
    if "documents" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("documents")}
    legacy_columns = ["original_name", "sha256", "chunk_count", "indexed_at"]

    with engine.begin() as connection:
        for column_name in legacy_columns:
            if column_name in existing_columns:
                connection.execute(text(f"ALTER TABLE documents DROP COLUMN {column_name}"))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _migrate_documents_table()
    _migrate_ingested_documents_data()
    _migrate_ingested_documents_table()
    _cleanup_documents_table()
