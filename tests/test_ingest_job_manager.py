import unittest

from app.services import ingest_job_manager as ingest_job_manager_module
from app.services.ingest_job_manager import IngestJobManager


class DummyRAGService:
    def __init__(self) -> None:
        self.invalidated = False

    def invalidate_cache(self) -> None:
        self.invalidated = True


class IngestJobManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_get_ingestion_service = ingest_job_manager_module.get_ingestion_service
        self.previous_get_rag_service = ingest_job_manager_module.get_rag_service
        self.addCleanup(self._restore_dependencies)

    def _restore_dependencies(self) -> None:
        ingest_job_manager_module.get_ingestion_service = self.previous_get_ingestion_service
        ingest_job_manager_module.get_rag_service = self.previous_get_rag_service

    def test_run_job_invalidates_rag_cache_after_success(self) -> None:
        manager = IngestJobManager()
        dummy_rag = DummyRAGService()

        class DummyIngestionService:
            @staticmethod
            def upload_all(**kwargs):
                del kwargs
                return {
                    "running": False,
                    "status": "completed",
                    "message": "Ingest selesai.",
                    "force": False,
                    "target_files": ["manual.pdf"],
                    "uploaded": 1,
                    "skipped": 0,
                    "failed": 0,
                    "failed_files": [],
                    "indexed_files": 1,
                    "total_files": 1,
                    "processed_files": 1,
                    "current_file": "",
                }

        ingest_job_manager_module.get_ingestion_service = lambda: DummyIngestionService()
        ingest_job_manager_module.get_rag_service = lambda: dummy_rag

        manager._run_job(force=False, source_files=["manual.pdf"])

        self.assertTrue(dummy_rag.invalidated)
        self.assertEqual(manager.get_status()["status"], "completed")

    def test_run_job_invalidates_rag_cache_after_failure(self) -> None:
        manager = IngestJobManager()
        dummy_rag = DummyRAGService()

        class DummyIngestionService:
            @staticmethod
            def upload_all(**kwargs):
                del kwargs
                raise RuntimeError("boom")

        ingest_job_manager_module.get_ingestion_service = lambda: DummyIngestionService()
        ingest_job_manager_module.get_rag_service = lambda: dummy_rag

        manager._run_job(force=True, source_files=["manual.pdf"])

        self.assertTrue(dummy_rag.invalidated)
        self.assertEqual(manager.get_status()["status"], "failed")


if __name__ == "__main__":
    unittest.main()
