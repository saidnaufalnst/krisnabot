import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import admin as admin_module
from app.api.admin import router as admin_router
from app.core.config import settings


class DummyIngestionService:
    def __init__(self) -> None:
        self.docs = ["existing.pdf"]

    def save_files(self, files):
        saved = []
        for name, _content in files:
            saved.append(name)
            self.docs.append(name)
        return saved

    def list_docs(self) -> list[str]:
        return list(self.docs)

    def list_indexed_files(self) -> list[str]:
        return []


class AdminApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.override_setting("environment", "development")
        self.override_setting("krisnabot_service_key", "")
        self.dummy_service = DummyIngestionService()
        self.previous_get_ingestion_service = admin_module.get_ingestion_service
        admin_module.get_ingestion_service = lambda: self.dummy_service
        self.addCleanup(self._restore_ingestion_service)

        app = FastAPI()
        app.include_router(admin_router)
        self.client = TestClient(app)

    def override_setting(self, name: str, value) -> None:
        previous = getattr(settings, name)
        object.__setattr__(settings, name, value)
        self.addCleanup(object.__setattr__, settings, name, previous)

    def _restore_ingestion_service(self) -> None:
        admin_module.get_ingestion_service = self.previous_get_ingestion_service

    def test_upload_accepts_plural_files_field(self) -> None:
        response = self.client.post(
            "/admin/upload",
            files=[("files", ("manual.pdf", b"%PDF-1.4", "application/pdf"))],
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "saved_files": ["manual.pdf"],
                "total_files_in_docs": 2,
            },
        )

    def test_upload_accepts_singular_file_field(self) -> None:
        response = self.client.post(
            "/admin/upload",
            files=[("file", ("manual.pdf", b"%PDF-1.4", "application/pdf"))],
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "saved_files": ["manual.pdf"],
                "total_files_in_docs": 2,
            },
        )

    def test_upload_returns_clear_error_when_no_file_field_is_sent(self) -> None:
        response = self.client.post("/admin/upload")

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json(),
            {
                "detail": "Field upload wajib diisi pada multipart form dengan nama 'files' atau 'file'.",
            },
        )


if __name__ == "__main__":
    unittest.main()
