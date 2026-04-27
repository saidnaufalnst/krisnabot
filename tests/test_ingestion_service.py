import unittest
from unittest.mock import patch

from app.services.ingestion_service import IngestionService, get_ingestion_service


class IngestionServiceLifecycleTests(unittest.TestCase):
    def tearDown(self) -> None:
        get_ingestion_service.cache_clear()

    def test_init_does_not_run_cleanup_implicitly(self) -> None:
        with patch.object(IngestionService, "_cleanup_orphaned_index_entries") as cleanup_index:
            with patch.object(IngestionService, "_cleanup_orphaned_checkpoints") as cleanup_checkpoints:
                service = IngestionService()
                cleanup_index.assert_not_called()
                cleanup_checkpoints.assert_not_called()

                service.run_startup_cleanup()

                cleanup_index.assert_called_once_with()
                cleanup_checkpoints.assert_called_once_with()

    def test_get_ingestion_service_returns_cached_instance(self) -> None:
        first = get_ingestion_service()
        second = get_ingestion_service()

        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
