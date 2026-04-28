import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.config import settings
from app.services.gemini_file_search_service import GeminiFileSearchService


class GeminiFileSearchServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = GeminiFileSearchService.__new__(GeminiFileSearchService)

    def override_setting(self, name: str, value) -> None:
        previous = getattr(settings, name)
        object.__setattr__(settings, name, value)
        self.addCleanup(object.__setattr__, settings, name, previous)

    def test_wait_for_document_active_polls_until_active(self) -> None:
        self.override_setting("file_search_document_poll_interval_seconds", 0.2)
        self.override_setting("file_search_document_ready_timeout_seconds", 5)

        calls: list[str] = []
        states = [
            SimpleNamespace(state="STATE_PENDING"),
            SimpleNamespace(state="STATE_ACTIVE"),
        ]

        def fake_get(*, name: str):
            calls.append(name)
            return states.pop(0)

        self.service.client = SimpleNamespace(
            file_search_stores=SimpleNamespace(
                documents=SimpleNamespace(get=fake_get),
            )
        )

        with patch("app.services.gemini_file_search_service.time.sleep", lambda _: None):
            document = self.service._wait_for_document_active("fileSearchStores/store/documents/manual")

        self.assertEqual(document.state, "STATE_ACTIVE")
        self.assertEqual(calls, ["fileSearchStores/store/documents/manual", "fileSearchStores/store/documents/manual"])

    def test_wait_for_document_active_raises_when_document_failed(self) -> None:
        self.service.client = SimpleNamespace(
            file_search_stores=SimpleNamespace(
                documents=SimpleNamespace(
                    get=lambda *, name: SimpleNamespace(name=name, state="STATE_FAILED")
                ),
            )
        )

        with self.assertRaisesRegex(RuntimeError, "gagal diproses"):
            self.service._wait_for_document_active("fileSearchStores/store/documents/manual")

    def test_wait_for_document_active_accepts_enum_like_state_string(self) -> None:
        self.override_setting("file_search_document_poll_interval_seconds", 0.2)
        self.override_setting("file_search_document_ready_timeout_seconds", 5)

        self.service.client = SimpleNamespace(
            file_search_stores=SimpleNamespace(
                documents=SimpleNamespace(
                    get=lambda *, name: SimpleNamespace(name=name, state="DocumentState.STATE_ACTIVE")
                ),
            )
        )

        with patch("app.services.gemini_file_search_service.time.sleep", lambda _: None):
            document = self.service._wait_for_document_active("fileSearchStores/store/documents/manual")

        self.assertEqual(document.state, "DocumentState.STATE_ACTIVE")

    def test_document_name_from_operation_response_expands_short_document_id(self) -> None:
        response = SimpleNamespace(
            parent="fileSearchStores/store",
            document_name="manual-123",
        )

        document_name = self.service._document_name_from_operation_response(
            response,
            "fileSearchStores/fallback",
        )

        self.assertEqual(document_name, "fileSearchStores/store/documents/manual-123")

    def test_upload_document_uses_direct_upload_to_file_search_store(self) -> None:
        uploaded: dict[str, object] = {}
        waited: list[str] = []

        def fake_upload_to_file_search_store(**kwargs):
            uploaded.update(kwargs)
            return SimpleNamespace(done=True)

        self.service.client = SimpleNamespace(
            file_search_stores=SimpleNamespace(
                upload_to_file_search_store=fake_upload_to_file_search_store,
            ),
        )
        self.service.get_or_create_store_name = lambda: "fileSearchStores/store"
        self.service._wait_for_operation = lambda operation: SimpleNamespace(
            response=SimpleNamespace(document_name="fileSearchStores/store/documents/manual-123")
        )
        self.service._wait_for_document_active = lambda document_name: waited.append(document_name)

        result = self.service.upload_document(
            source_file="manual.pdf",
            content=b"pdf-content",
            mime_type="application/pdf",
            sha256="abc123",
        )

        self.assertEqual(result["store_name"], "fileSearchStores/store")
        self.assertEqual(result["document_name"], "fileSearchStores/store/documents/manual-123")
        self.assertEqual(waited, ["fileSearchStores/store/documents/manual-123"])
        self.assertEqual(uploaded["config"].display_name, "manual.pdf")
        self.assertEqual(uploaded["config"].mime_type, "application/pdf")
        self.assertFalse(Path(str(uploaded["file"])).exists())
        self.assertEqual(uploaded["file_search_store_name"], "fileSearchStores/store")
        self.assertEqual(uploaded["config"].custom_metadata[0].key, "source_file")
        self.assertEqual(uploaded["config"].custom_metadata[0].string_value, "manual.pdf")
        self.assertEqual(uploaded["config"].custom_metadata[1].key, "sha256")
        self.assertEqual(uploaded["config"].custom_metadata[1].string_value, "abc123")


if __name__ == "__main__":
    unittest.main()
