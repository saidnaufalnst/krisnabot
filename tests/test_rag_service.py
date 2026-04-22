import unittest

from app.core.config import settings
from app.services.rag_service import RAGService


class RAGServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rag = RAGService()
        self.rag._log_chat = lambda *args, **kwargs: None

    def override_setting(self, name: str, value) -> None:
        previous = getattr(settings, name)
        object.__setattr__(settings, name, value)
        self.addCleanup(object.__setattr__, settings, name, previous)

    def test_detect_social_response_returns_greeting(self) -> None:
        result = self.rag._detect_social_response("halo")

        self.assertIsNotNone(result)
        self.assertTrue(result["found"])
        self.assertIn("KRISNABOT", result["answer"])

    def test_ask_returns_context_not_found_when_no_indexed_files(self) -> None:
        self.rag.ingestion = type(
            "DummyIngestion",
            (),
            {"list_indexed_files": staticmethod(lambda: [])},
        )()

        result = self.rag.ask("Apa itu KRISNA?")

        self.assertFalse(result["found"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["error"], "context_not_found")
        self.assertEqual(result["used_files"], [])

    def test_ask_uses_file_search_result_and_preserves_used_files(self) -> None:
        self.rag.ingestion = type(
            "DummyIngestion",
            (),
            {"list_indexed_files": staticmethod(lambda: ["Manual.pdf"])},
        )()
        self.rag._generate_answer_with_file_search = lambda prompt: {
            "found": True,
            "answer": "KRISNA adalah aplikasi perencanaan.",
            "message": "",
            "error": "",
            "used_files": ["Manual.pdf"],
        }

        result = self.rag.ask("Apa itu KRISNA?")

        self.assertTrue(result["found"])
        self.assertEqual(result["answer"], "KRISNA adalah aplikasi perencanaan.")
        self.assertEqual(result["message"], "")
        self.assertEqual(result["used_files"], ["Manual.pdf"])

    def test_ask_does_not_retry_when_retry_disabled(self) -> None:
        self.override_setting("chat_retry_on_empty_answer", False)
        self.rag.ingestion = type(
            "DummyIngestion",
            (),
            {"list_indexed_files": staticmethod(lambda: ["Manual.pdf"])},
        )()

        calls: list[str] = []

        def fake_generate(prompt: str):
            calls.append(prompt)
            return {
                "found": False,
                "answer": "",
                "message": "Saya belum menemukan jawaban yang cukup jelas.",
                "error": "empty_model_text",
                "used_files": ["Manual.pdf"],
            }

        self.rag._generate_answer_with_file_search = fake_generate

        result = self.rag.ask("Apa itu KRISNA?")

        self.assertEqual(len(calls), 1)
        self.assertFalse(result["found"])
        self.assertEqual(result["error"], "empty_model_text")
        self.assertEqual(result["used_files"], [])

    def test_ask_retries_once_when_answer_empty_but_used_files_present(self) -> None:
        self.override_setting("chat_retry_on_empty_answer", True)
        self.rag.ingestion = type(
            "DummyIngestion",
            (),
            {"list_indexed_files": staticmethod(lambda: ["Manual.pdf"])},
        )()

        calls: list[str] = []

        def fake_generate(prompt: str):
            calls.append(prompt)
            if len(calls) == 1:
                return {
                    "found": False,
                    "answer": "",
                    "message": "Saya belum menemukan jawaban yang cukup jelas.",
                    "error": "empty_model_text",
                    "used_files": ["Manual.pdf"],
                }
            return {
                "found": True,
                "answer": "KRISNA adalah aplikasi perencanaan dan penganggaran.",
                "message": "",
                "error": "",
                "used_files": ["Manual.pdf"],
            }

        self.rag._generate_answer_with_file_search = fake_generate

        result = self.rag.ask("Apa itu KRISNA?")

        self.assertEqual(len(calls), 2)
        self.assertTrue(result["found"])
        self.assertEqual(result["answer"], "KRISNA adalah aplikasi perencanaan dan penganggaran.")
        self.assertEqual(result["used_files"], ["Manual.pdf"])

    def test_ask_returns_out_of_scope_for_non_krisna_question(self) -> None:
        result = self.rag.ask("Siapa presiden Indonesia saat ini?")

        self.assertFalse(result["found"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["error"], "out_of_scope_question")
        self.assertEqual(result["used_files"], [])

    def test_ask_routes_unanswered_technical_question_to_helpdesk(self) -> None:
        self.rag.ingestion = type(
            "DummyIngestion",
            (),
            {"list_indexed_files": staticmethod(lambda: ["Manual.pdf"])},
        )()
        self.rag._generate_answer_with_file_search = lambda prompt: {
            "found": False,
            "answer": "",
            "message": "Saya belum menemukan rujukan yang cukup relevan pada dokumen yang tersedia.",
            "error": "context_not_found",
            "used_files": [],
        }

        result = self.rag.ask("Kenapa login KRISNA error 403?")

        self.assertFalse(result["found"])
        self.assertEqual(result["error"], "technical_help_required")
        self.assertIn("helpdesk", result["message"])

    def test_generate_answer_with_file_search_uses_fallback_model_on_retryable_error(self) -> None:
        self.override_setting("file_search_model_name", "primary-model")
        self.override_setting("file_search_fallback_model_name", "fallback-model")
        self.override_setting("model_name", "primary-model")
        self.rag.file_search_service = type(
            "DummyFileSearchService",
            (),
            {
                "extract_used_files": staticmethod(lambda response: ["Manual.pdf"]),
                "has_grounding": staticmethod(lambda response: True),
            },
        )()

        calls: list[str] = []

        def fake_call(prompt: str, model_name: str):
            del prompt
            calls.append(model_name)
            if model_name == "primary-model":
                raise Exception(
                    "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand.'}}"
                )
            return type("Response", (), {"text": "Jawaban dari fallback."})()

        self.rag._call_file_search_model = fake_call

        result = self.rag._generate_answer_with_file_search("Apa itu KRISNA?")

        self.assertEqual(calls, ["primary-model", "fallback-model"])
        self.assertTrue(result["found"])
        self.assertEqual(result["answer"], "Jawaban dari fallback.")
        self.assertEqual(result["used_files"], ["Manual.pdf"])


if __name__ == "__main__":
    unittest.main()
