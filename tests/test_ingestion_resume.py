import unittest
from types import SimpleNamespace

from app.services.ingestion_service import IngestionService


class DummyCheckpointRepository:
    def __init__(self):
        self.record = None
        self.upserts = []
        self.cleared = []

    def list_all(self):
        return []

    def get(self, source_file):
        if self.record is None or self.record.source_file != source_file:
            return None
        return self.record

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)
        self.record = SimpleNamespace(**kwargs)

    def clear(self, source_file):
        self.cleared.append(source_file)
        if self.record is not None and self.record.source_file == source_file:
            self.record = None


class DummyIngestedDocumentRepository:
    def __init__(self, existing_record=None):
        self.record = existing_record
        self.entries = []
        self.cleared = []

    def list_all(self):
        return []

    def get(self, source_file):
        if self.record is None or self.record.source_file != source_file:
            return None
        return self.record

    def upsert(self, **kwargs):
        self.entries.append(kwargs)
        self.record = SimpleNamespace(**kwargs)

    def clear(self, source_file):
        self.cleared.append(source_file)
        self.record = None


class DummyFileSearchService:
    def __init__(self, fail_on_upload=False):
        self.fail_on_upload = fail_on_upload
        self.upload_calls = []
        self.deleted_documents = []
        self.deleted_by_source = []

    def upload_document(self, *, source_file, content, mime_type, sha256=None):
        self.upload_calls.append(
            {
                "source_file": source_file,
                "content": content,
                "mime_type": mime_type,
                "sha256": sha256,
            }
        )
        if self.fail_on_upload:
            raise RuntimeError("file search upload failed")
        return {
            "store_name": "fileSearchStores/krisnabot-store",
            "document_name": f"fileSearchStores/krisnabot-store/documents/{source_file}",
        }

    def delete_document(self, document_name):
        self.deleted_documents.append(document_name)

    def delete_document_by_source_file(self, source_file, *, store_name=None):
        self.deleted_by_source.append({"source_file": source_file, "store_name": store_name})


class HybridIngestionTests(unittest.TestCase):
    def _build_service(self, *, existing_record=None, fail_on_upload=False):
        service = IngestionService.__new__(IngestionService)
        service.document_repository = SimpleNamespace()
        service.ingested_document_repository = DummyIngestedDocumentRepository(existing_record=existing_record)
        service.ingest_checkpoint_repository = DummyCheckpointRepository()
        service.audit_repository = SimpleNamespace()
        service._file_search_service = DummyFileSearchService(fail_on_upload=fail_on_upload)
        service._sha256_bytes = lambda content: "same-hash"
        return service

    def test_index_pdf_persists_failed_checkpoint_when_file_search_upload_fails(self):
        service = self._build_service(fail_on_upload=True)

        with self.assertRaisesRegex(RuntimeError, "file search upload failed"):
            service._index_pdf("manual.pdf", b"dummy-content")

        checkpoint = service.ingest_checkpoint_repository.record
        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint.source_file, "manual.pdf")
        self.assertEqual(checkpoint.sha256, "same-hash")
        self.assertEqual(checkpoint.total_chunks, 1)
        self.assertEqual(checkpoint.completed_chunks, 0)
        self.assertEqual(checkpoint.status, "failed")
        self.assertEqual(service.ingested_document_repository.entries, [])

    def test_index_pdf_persists_remote_document_metadata_after_success(self):
        service = self._build_service()
        progress_events = []

        result = service._index_pdf(
            "manual.pdf",
            b"dummy-content",
            progress_callback=progress_events.append,
        )

        self.assertEqual(result["chunk_count"], 1)
        self.assertEqual(result["file_search_store_name"], "fileSearchStores/krisnabot-store")
        self.assertEqual(
            result["remote_document_name"],
            "fileSearchStores/krisnabot-store/documents/manual.pdf",
        )
        self.assertEqual(len(service.ingested_document_repository.entries), 1)
        self.assertEqual(
            service.ingested_document_repository.entries[0]["remote_document_name"],
            "fileSearchStores/krisnabot-store/documents/manual.pdf",
        )
        self.assertEqual(service.ingest_checkpoint_repository.cleared, ["manual.pdf"])
        self.assertEqual(progress_events[0]["current_chunk"], 0)
        self.assertEqual(progress_events[-1]["current_chunk"], 1)

    def test_index_pdf_deletes_previous_remote_document_before_reupload(self):
        existing_record = SimpleNamespace(
            source_file="manual.pdf",
            remote_document_name="fileSearchStores/krisnabot-store/documents/old-manual",
            file_search_store_name="fileSearchStores/krisnabot-store",
        )
        service = self._build_service(existing_record=existing_record)

        service._index_pdf("manual.pdf", b"dummy-content")

        self.assertEqual(
            service._file_search_service.deleted_documents,
            ["fileSearchStores/krisnabot-store/documents/old-manual"],
        )
        self.assertEqual(len(service._file_search_service.upload_calls), 1)


if __name__ == "__main__":
    unittest.main()
