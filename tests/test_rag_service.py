import unittest

from google.genai.errors import ServerError

from app.core.config import settings
from app.services.gemini_file_search_service import GeminiFileSearchService
from app.services.rag_service import RAGService, SYSTEM_INSTRUCTION


class RAGServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rag = RAGService()
        self.rag._log_chat = lambda *args, **kwargs: None
        self.rag._retry_backoff_seconds = 0

    def override_setting(self, name: str, value) -> None:
        previous = getattr(settings, name)
        object.__setattr__(settings, name, value)
        self.addCleanup(object.__setattr__, settings, name, previous)

    def test_detect_social_returns_greeting(self) -> None:
        result = self.rag._detect_social("halo")

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
        self.assertIn("helpdesk", result["message"])
        self.assertEqual(result["used_files"], [])

    def test_ask_uses_file_search_result_and_preserves_used_files(self) -> None:
        self.rag.ingestion = type(
            "DummyIngestion",
            (),
            {"list_indexed_files": staticmethod(lambda: ["Manual.pdf"])},
        )()
        self.rag._generate_answer = lambda prompt: {
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

    def test_ask_does_not_retry_when_answer_empty(self) -> None:
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

        self.rag._generate_answer = fake_generate

        result = self.rag.ask("Apa itu KRISNA?")

        self.assertEqual(len(calls), 1)
        self.assertFalse(result["found"])
        self.assertEqual(result["error"], "empty_model_text")
        self.assertIn("helpdesk", result["message"])
        self.assertEqual(result["used_files"], [])

    def test_ask_returns_empty_model_text_without_second_generation(self) -> None:
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

        self.rag._generate_answer = fake_generate

        result = self.rag.ask("Apa itu KRISNA?")

        self.assertEqual(len(calls), 1)
        self.assertFalse(result["found"])
        self.assertEqual(result["error"], "empty_model_text")
        self.assertIn("helpdesk", result["message"])
        self.assertEqual(result["used_files"], [])

    def test_ask_retries_once_with_recovery_prompt_when_grounding_missing(self) -> None:
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
                    "message": "Saya belum bisa memastikan rujukan jawaban dari dokumen, jadi jawaban tidak saya tampilkan.",
                    "error": "missing_grounding_metadata",
                    "used_files": [],
                }
            return {
                "found": True,
                "answer": "Untuk menambah Rincian Output, buka menu terkait lalu klik Tambah Data.",
                "message": "",
                "error": "",
                "used_files": ["Manual.pdf"],
            }

        self.rag._generate_answer = fake_generate

        result = self.rag.ask("Saya tidak bisa menambah rincian output")

        self.assertEqual(len(calls), 2)
        self.assertIn("Saya tidak bisa menambah rincian output", calls[0])
        self.assertIn("Pertanyaan pengguna:", calls[1])
        self.assertIn("Pertanyaan untuk pencarian ulang:", calls[1])
        self.assertIn("Bagaimana cara menambah rincian output?", calls[1])
        self.assertTrue(result["found"])
        self.assertEqual(
            result["answer"],
            "Untuk menambah Rincian Output, buka menu terkait lalu klik Tambah Data.",
        )
        self.assertEqual(result["used_files"], ["Manual.pdf"])

    def test_ask_retries_with_recovery_prompt_for_context_not_found_complaint(self) -> None:
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
                    "message": "Saya belum menemukan rujukan yang cukup relevan pada dokumen yang tersedia.",
                    "error": "context_not_found",
                    "used_files": [],
                }
            return {
                "found": True,
                "answer": "Periksa hak akses login dan pastikan akun Anda aktif.",
                "message": "",
                "error": "",
                "used_files": ["Manual.pdf"],
            }

        self.rag._generate_answer = fake_generate

        result = self.rag.ask("Kenapa login KRISNA error 403?")

        self.assertEqual(len(calls), 2)
        self.assertIn("Pertanyaan pengguna:", calls[1])
        self.assertTrue(result["found"])
        self.assertEqual(result["used_files"], ["Manual.pdf"])

    def test_ask_does_not_retry_for_definition_context_not_found(self) -> None:
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
                "message": "Saya belum menemukan rujukan yang cukup relevan pada dokumen yang tersedia.",
                "error": "context_not_found",
                "used_files": [],
            }

        self.rag._generate_answer = fake_generate

        result = self.rag.ask("Apa itu KRISNA?")

        self.assertEqual(len(calls), 1)
        self.assertFalse(result["found"])
        self.assertEqual(result["error"], "context_not_found")

    def test_ask_preserves_context_not_found_for_unanswered_question(self) -> None:
        self.rag.ingestion = type(
            "DummyIngestion",
            (),
            {"list_indexed_files": staticmethod(lambda: ["Manual.pdf"])},
        )()
        self.rag._generate_answer = lambda prompt: {
            "found": False,
            "answer": "",
            "message": "Saya belum menemukan rujukan yang cukup relevan pada dokumen yang tersedia.",
            "error": "context_not_found",
            "used_files": [],
        }

        result = self.rag.ask("Kenapa login KRISNA error 403?")

        self.assertFalse(result["found"])
        self.assertEqual(result["error"], "context_not_found")
        self.assertIn("helpdesk", result["message"])

    def test_ask_returns_greeting_for_random_input(self) -> None:
        result = self.rag.ask("tes")

        self.assertTrue(result["found"])
        self.assertIn("KRISNABOT", result["answer"])
        self.assertEqual(result["message"], "")
        self.assertEqual(result["used_files"], [])
        self.assertEqual(result["grounding_chunks"], [])

    def test_ask_returns_greeting_for_single_character_input(self) -> None:
        result = self.rag.ask("p")

        self.assertTrue(result["found"])
        self.assertEqual(result["answer"], "Halo, saya KRISNABOT. Silakan ajukan pertanyaan terkait KRISNA.")

    def test_ask_returns_greeting_for_casual_phrase(self) -> None:
        result = self.rag.ask("Apa kabar")

        self.assertTrue(result["found"])
        self.assertEqual(result["answer"], "Halo, saya KRISNABOT. Silakan ajukan pertanyaan terkait KRISNA.")
        self.assertEqual(result["message"], "")

    def test_ask_redirects_grounded_unanswered_krisna_answer_to_helpdesk(self) -> None:
        self.rag.ingestion = type(
            "DummyIngestion",
            (),
            {"list_indexed_files": staticmethod(lambda: ["Manual.pdf"])},
        )()
        self.rag._generate_answer = lambda prompt: {
            "found": True,
            "answer": "Jawaban belum tersedia di dokumen KRISNA.",
            "message": "",
            "error": "",
            "used_files": ["Manual.pdf"],
        }

        result = self.rag.ask("Apa syarat khusus di KRISNA?")

        self.assertFalse(result["found"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["error"], "context_not_found")
        self.assertIn("helpdesk", result["message"])
        self.assertEqual(result["used_files"], [])

    def test_ask_redirects_related_object_substitution_to_helpdesk(self) -> None:
        self.rag.ingestion = type(
            "DummyIngestion",
            (),
            {"list_indexed_files": staticmethod(lambda: ["Manual.pdf"])},
        )()
        self.rag._generate_answer = lambda prompt: {
            "found": True,
            "answer": (
                "Dokumen rujukan tidak secara spesifik menampilkan langkah Rincian Output, "
                "namun menjelaskan Klasifikasi Rincian Output yang merupakan wadah untuk Rincian Output."
            ),
            "message": "",
            "error": "",
            "used_files": ["Manual.pdf"],
        }

        result = self.rag.ask("Bagaimana menambah Rincian Output?")

        self.assertFalse(result["found"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["error"], "context_not_found")
        self.assertIn("helpdesk", result["message"])
        self.assertEqual(result["used_files"], [])

    def test_system_instruction_requests_detailed_answers_and_notes(self) -> None:
        self.assertIn("lengkap dan detail sesuai isi rujukan", SYSTEM_INSTRUCTION)
        self.assertIn("SEMUA langkah dari awal hingga akhir", SYSTEM_INSTRUCTION)
        self.assertIn("Jangan mengawali jawaban dengan frasa 'Berdasarkan dokumen yang tersedia,'", SYSTEM_INSTRUCTION)
        self.assertIn("diawali **Catatan:**", SYSTEM_INSTRUCTION)
        self.assertIn("Letakkan catatan setelah uraian", SYSTEM_INSTRUCTION)
        self.assertIn("kumpulkan catatan tersebut berurutan di akhir bagian", SYSTEM_INSTRUCTION)
        self.assertIn("jangan menambahkan catatan fiktif", SYSTEM_INSTRUCTION)

    def test_build_prompt_requests_detailed_file_search_answer(self) -> None:
        prompt = self.rag._build_prompt("Apa itu KRISNA?")

        self.assertIn("Apa itu KRISNA?", prompt)
        self.assertIn("Jawab langsung berdasarkan rujukan File Search yang relevan", prompt)
        self.assertIn("Jangan awali jawaban dengan frasa 'Berdasarkan dokumen yang tersedia,'", prompt)
        self.assertIn("lengkap dan detail", prompt)
        self.assertIn("SEMUA langkah secara berurutan", prompt)
        self.assertIn("Tambahkan **Catatan:**", prompt)
        self.assertIn("Letakkan catatan setelah uraian", prompt)
        self.assertIn("kumpulkan catatan tersebut berurutan di akhir bagian", prompt)
        self.assertIn("tidak bisa/gagal/error/kendala", prompt)
        self.assertIn("cara melakukan hal yang dimaksud", prompt)
        self.assertIn("prosedur objek induk", prompt)
        self.assertIn("Jika rujukan tidak cukup", prompt)
        self.assertNotIn("rujukan lintas dokumen", prompt)
        self.assertNotIn("manual dahulu", prompt)
        self.assertNotIn("Istilah terkait untuk pencarian", prompt)

    def test_build_prompt_uses_generic_object_matching_not_hardcoded_ro_guard(self) -> None:
        prompt = self.rag._build_prompt("Bagaimana cara menambah Rincian Output?")

        self.assertIn("Bagaimana cara menambah Rincian Output?", prompt)
        self.assertIn("File Search", prompt)
        self.assertNotIn("RO berbeda dari Klasifikasi Rincian Output", prompt)
        self.assertNotIn("Copy KRO", prompt)

    def test_build_recovery_prompt_is_generic_not_domain_hardcoded(self) -> None:
        prompt = self.rag._build_recovery_prompt("Saya tidak bisa menambah Rincian Output")

        self.assertIn("Pertanyaan untuk pencarian ulang:", prompt)
        self.assertIn("Bagaimana cara menambah Rincian Output?", prompt)
        self.assertIn("Jangan awali jawaban dengan frasa 'Berdasarkan dokumen yang tersedia,'", prompt)
        self.assertIn("lengkap dan detail", prompt)
        self.assertIn("SEMUA langkah yang ada di rujukan", prompt)
        self.assertIn("Tambahkan **Catatan:**", prompt)
        self.assertIn("Letakkan catatan setelah uraian", prompt)
        self.assertIn("kumpulkan catatan tersebut berurutan di akhir bagian", prompt)
        self.assertIn("Utamakan objek yang tertulis", prompt)
        self.assertIn("prosedur objek induk", prompt)
        self.assertIn("Jika rujukan hanya memuat objek terkait", prompt)
        self.assertNotIn("RO berbeda dari Klasifikasi Rincian Output", prompt)
        self.assertNotIn("Copy KRO", prompt)

    def test_build_recovery_question_turns_complaint_into_how_to_question(self) -> None:
        self.assertEqual(
            self.rag._build_recovery_question("Saya tidak bisa menambah Rincian Output"),
            "Bagaimana cara menambah Rincian Output?",
        )
        self.assertEqual(
            self.rag._build_recovery_question("Gagal upload dokumen"),
            "Bagaimana cara upload dokumen?",
        )

    def test_get_store_names_skips_known_store_lookup_when_no_index_exists(self) -> None:
        self.rag.ingestion = type(
            "DummyIngestion",
            (),
            {
                "list_file_search_store_names": staticmethod(lambda: []),
                "list_indexed_files": staticmethod(lambda: []),
            },
        )()
        self.rag.file_search_service = type(
            "DummyFileSearchService",
            (),
            {
                "get_known_store_name": staticmethod(
                    lambda: (_ for _ in ()).throw(AssertionError("known store lookup should not be called"))
                )
            },
        )()

        self.assertEqual(self.rag._get_store_names(), [])

    def test_get_store_names_uses_known_store_when_index_exists(self) -> None:
        self.rag.ingestion = type(
            "DummyIngestion",
            (),
            {
                "list_file_search_store_names": staticmethod(lambda: []),
                "list_indexed_files": staticmethod(lambda: ["Manual.pdf"]),
            },
        )()
        self.rag.file_search_service = type(
            "DummyFileSearchService",
            (),
            {"get_known_store_name": staticmethod(lambda: "fileSearchStores/test")},
        )()

        self.assertEqual(self.rag._get_store_names(), ["fileSearchStores/test"])

    def test_clean_answer_preserves_simple_markdown(self) -> None:
        raw_answer = (
            "1. **Akses Halaman:** Pilih menu `RKP`, lalu klik *Tambah Data*.\n"
            "* Hak Akses: Admin mengatur role *viewer* dan *submit*.\n\n"
            "Catatan: * Jika data belum muncul, hubungi admin."
        )

        cleaned = self.rag._clean_answer(raw_answer)

        self.assertEqual(
            cleaned,
            (
                "1. **Akses Halaman:** Pilih menu `RKP`, lalu klik *Tambah Data*.\n"
                "- Hak Akses: Admin mengatur role *viewer* dan *submit*.\n\n"
                "**Catatan:** Jika data belum muncul, hubungi admin."
            ),
        )

    def test_clean_answer_bolds_catatan_penting_label(self) -> None:
        raw_answer = (
            "Catatan penting:\n"
            "* Kode KRO dan nomenklatur akan terisi otomatis."
        )

        cleaned = self.rag._clean_answer(raw_answer)

        self.assertEqual(
            cleaned,
            (
                "**Catatan penting:**\n"
                "- Kode KRO dan nomenklatur akan terisi otomatis."
            ),
        )

    def test_clean_answer_removes_exact_available_document_preamble(self) -> None:
        raw_answer = "Berdasarkan dokumen yang tersedia, terdapat tiga cara untuk menambah RO."

        cleaned = self.rag._clean_answer(raw_answer)

        self.assertEqual(cleaned, "Terdapat tiga cara untuk menambah RO.")

    def test_clean_answer_removes_information_and_procedure_labels(self) -> None:
        raw_answer = (
            "Informasi: Penambahan nomenklatur kegiatan tidak dapat dilakukan secara mandiri.\n\n"
            "Prosedur:\n"
            "1. Pimpinan K/L mengirimkan surat permohonan perubahan kegiatan.\n"
            "2. Admin Pusat melakukan input data setelah mendapat persetujuan."
        )

        cleaned = self.rag._clean_answer(raw_answer)

        self.assertEqual(
            cleaned,
            (
                "Penambahan nomenklatur kegiatan tidak dapat dilakukan secara mandiri.\n\n"
                "1. Pimpinan K/L mengirimkan surat permohonan perubahan kegiatan.\n"
                "2. Admin Pusat melakukan input data setelah mendapat persetujuan."
            ),
        )

    def test_clean_answer_trims_incomplete_tail(self) -> None:
        raw_answer = (
            "1. **Akses Menu:** Pilih menu RKP.\n"
            "2. **Tambah Data:** Klik tombol **Tambah Data**.\n\n"
            "**Catatan:** Data akan berstatus `pending-add`. "
            "Pengguna yang berwenang adalah PJ RKP, PJ PN, atau"
        )

        cleaned = self.rag._clean_answer(raw_answer)

        self.assertEqual(
            cleaned,
            (
                "1. **Akses Menu:** Pilih menu RKP.\n"
                "2. **Tambah Data:** Klik tombol **Tambah Data**.\n\n"
                "**Catatan:** Data akan berstatus `pending-add`."
            ),
        )

    def test_clean_answer_trims_dangling_list_marker(self) -> None:
        raw_answer = (
            "Berdasarkan dokumen KRISNA, terdapat dua cara yang dijelaskan terkait penambahan RO:\n\n"
            "1."
        )

        cleaned = self.rag._clean_answer(raw_answer)

        self.assertEqual(
            cleaned,
            "Berdasarkan dokumen KRISNA, terdapat dua cara yang dijelaskan terkait penambahan RO:",
        )

    def test_clean_answer_removes_duplicate_cited_tail(self) -> None:
        raw_answer = (
            "Untuk menambah Rincian Output (RO) di KRISNA, Anda dapat melakukannya dengan beberapa cara:\n\n"
            "**Menambah Data Rincian Output Secara Manual:**\n"
            "Pada halaman tabel Klasifikasi Rincian Output, lakukan *drill down*.\n"
            "Setelah masuk ke halaman Rincian Output, klik tombol Tambah Data.\n"
            "Akan muncul formulir Tambah Data.\n\n"
            "[Untuk menambah Rincian Output (RO) di KRISNA, Anda dapat melakukannya dengan beberapa cara:\n"
            "**Menambah Data Rincian Output Secara Manual:**\n"
            "Pada halaman tabel Klasifikasi Rincian Output, lakukan *drill down*. [cite: 3]\n"
            "Setelah masuk ke halaman Rincian Output, klik tombol Tambah Data. [cite: 3]\n"
            "Akan muncul formulir Tambah Data. [cite: 3]"
        )

        cleaned = self.rag._clean_answer(raw_answer)

        self.assertEqual(
            cleaned,
            (
                "Untuk menambah Rincian Output (RO) di KRISNA, Anda dapat melakukannya dengan beberapa cara:\n\n"
                "**Menambah Data Rincian Output Secara Manual:**\n"
                "Pada halaman tabel Klasifikasi Rincian Output, lakukan *drill down*.\n"
                "Setelah masuk ke halaman Rincian Output, klik tombol Tambah Data.\n"
                "Akan muncul formulir Tambah Data."
            ),
        )

    def test_generate_answer_uses_model_name(self) -> None:
        self.override_setting("model_name", "configured-model")
        self.rag.file_search_service = type(
            "DummyFileSearchService",
            (),
            {
                "extract_grounding_chunks": staticmethod(
                    lambda response: [{"source_file": "Manual.pdf", "text": "Jawaban dari configured model."}]
                ),
                "extract_used_files": staticmethod(lambda response: ["Manual.pdf"]),
                "has_grounding": staticmethod(lambda response: True),
            },
        )()

        calls: list[str] = []

        def fake_generate_content(*, model: str, contents: str, config):
            del contents, config
            calls.append(model)
            return type("Response", (), {"text": "Jawaban dari configured model."})()

        self.rag._get_store_names = lambda: ["fileSearchStores/test"]
        self.rag.client = type(
            "DummyClient",
            (),
            {
                "models": type(
                    "DummyModels",
                    (),
                    {"generate_content": staticmethod(fake_generate_content)},
                )()
            },
        )()

        result = self.rag._generate_answer("Apa itu KRISNA?")

        self.assertEqual(calls, ["configured-model"])
        self.assertTrue(result["found"])
        self.assertEqual(result["answer"], "Jawaban dari configured model.")
        self.assertEqual(result["used_files"], ["Manual.pdf"])

    def test_file_search_service_extracts_grounding_chunk_text(self) -> None:
        service = GeminiFileSearchService.__new__(GeminiFileSearchService)
        retrieved_context = type("RetrievedContext", (), {"text": "Isi potongan sumber dari dokumen."})()
        chunk = type("GroundingChunk", (), {"retrieved_context": retrieved_context})()
        grounding_metadata = type("GroundingMetadata", (), {"grounding_chunks": [chunk]})()
        candidate = type("Candidate", (), {"grounding_metadata": grounding_metadata})()
        response = type("Response", (), {"candidates": [candidate]})()

        self.assertEqual(
            service.extract_grounding_chunks(response),
            [{"source_file": "", "text": "Isi potongan sumber dari dokumen."}],
        )
        self.assertTrue(service.has_grounding(response))

    def test_file_search_service_extracts_grounding_chunk_text_from_rest_shape(self) -> None:
        service = GeminiFileSearchService.__new__(GeminiFileSearchService)
        response = {
            "candidates": [
                {
                    "groundingMetadata": {
                        "groundingChunks": [
                            {
                                "retrievedContext": {
                                    "text": "Teks dari bentuk JSON REST.",
                                    "title": "Fallback.pdf",
                                    "customMetadata": [
                                        {"key": "source_file", "stringValue": "Manual.pdf"}
                                    ],
                                }
                            }
                        ]
                    }
                }
            ]
        }

        self.assertEqual(service.extract_used_files(response), ["Manual.pdf"])
        self.assertEqual(
            service.extract_grounding_chunks(response),
            [{"source_file": "Manual.pdf", "text": "Teks dari bentuk JSON REST."}],
        )
        self.assertTrue(service.has_grounding(response))

    def test_file_search_service_builds_chunking_config_from_settings(self) -> None:
        self.override_setting("file_search_max_tokens_per_chunk", 240)
        self.override_setting("file_search_max_overlap_tokens", 30)

        chunking_config = GeminiFileSearchService._chunking_config()

        self.assertIsNotNone(chunking_config)
        self.assertEqual(chunking_config.white_space_config.max_tokens_per_chunk, 240)
        self.assertEqual(chunking_config.white_space_config.max_overlap_tokens, 30)

    def test_file_search_service_omits_chunking_config_when_setting_zero(self) -> None:
        self.override_setting("file_search_max_tokens_per_chunk", 0)
        self.override_setting("file_search_max_overlap_tokens", 0)

        self.assertIsNone(GeminiFileSearchService._chunking_config())

    def test_call_model_omits_optional_overrides_when_config_zero(self) -> None:
        captured: dict[str, object] = {}
        self.rag._top_k = None
        self.rag._max_tokens = None

        def fake_generate_content(*, model: str, contents: str, config):
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config
            return type("Response", (), {"text": "ok"})()

        self.rag.client = type(
            "DummyClient",
            (),
            {
                "models": type(
                    "DummyModels",
                    (),
                    {"generate_content": staticmethod(fake_generate_content)},
                )()
            },
        )()

        self.rag._call_model("Pertanyaan", "configured-model", ["fileSearchStores/test"])

        config = captured["config"]
        file_search = config.tools[0].file_search
        self.assertEqual(captured["model"], "configured-model")
        self.assertEqual(captured["contents"], "Pertanyaan")
        self.assertIsNone(config.max_output_tokens)
        self.assertIsNone(file_search.top_k)
        self.assertEqual(file_search.file_search_store_names, ["fileSearchStores/test"])

    def test_call_model_sends_positive_optional_overrides(self) -> None:
        captured: dict[str, object] = {}
        self.rag._top_k = 15
        self.rag._max_tokens = 1200

        def fake_generate_content(*, model: str, contents: str, config):
            del model, contents
            captured["config"] = config
            return type("Response", (), {"text": "ok"})()

        self.rag.client = type(
            "DummyClient",
            (),
            {
                "models": type(
                    "DummyModels",
                    (),
                    {"generate_content": staticmethod(fake_generate_content)},
                )()
            },
        )()

        self.rag._call_model("Pertanyaan", "configured-model", ["fileSearchStores/test"])

        config = captured["config"]
        file_search = config.tools[0].file_search
        self.assertEqual(config.max_output_tokens, 1200)
        self.assertEqual(file_search.top_k, 15)

    def test_parse_response_accepts_grounding_chunk_text_without_source_file(self) -> None:
        self.rag.file_search_service = type(
            "DummyFileSearchService",
            (),
            {
                "extract_grounding_chunks": staticmethod(
                    lambda response: [{"source_file": "", "text": "Langkah menambah RO ada di menu Rincian Output."}]
                ),
                "extract_used_files": staticmethod(lambda response: []),
                "has_grounding": staticmethod(lambda response: True),
            },
        )()
        response = type("Response", (), {"text": "Untuk menambah RO, gunakan menu Rincian Output."})()

        result = self.rag._parse_response(response)

        self.assertTrue(result["found"])
        self.assertEqual(result["answer"], "Untuk menambah RO, gunakan menu Rincian Output.")
        self.assertEqual(result["used_files"], [])
        self.assertEqual(
            result["grounding_chunks"],
            [{"source_file": "", "text": "Langkah menambah RO ada di menu Rincian Output."}],
        )

    def test_parse_response_rejects_answer_without_grounding_metadata(self) -> None:
        self.rag.file_search_service = type(
            "DummyFileSearchService",
            (),
            {
                "extract_grounding_chunks": staticmethod(lambda response: []),
                "extract_used_files": staticmethod(lambda response: []),
                "has_grounding": staticmethod(lambda response: False),
            },
        )()
        response = type("Response", (), {"text": "Jawaban dari File Search tanpa metadata sumber."})()

        result = self.rag._parse_response(response)

        self.assertFalse(result["found"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["error"], "missing_grounding_metadata")
        self.assertEqual(result["used_files"], [])
        self.assertNotIn("lebih spesifik", result["message"])

    def test_parse_response_does_not_treat_search_queries_as_grounding(self) -> None:
        self.rag.file_search_service = type(
            "DummyFileSearchService",
            (),
            {
                "extract_grounding_chunks": staticmethod(lambda response: []),
                "extract_used_files": staticmethod(lambda response: []),
                "has_grounding": staticmethod(lambda response: False),
            },
        )()
        grounding_metadata = type(
            "GroundingMetadata",
            (),
            {
                "grounding_chunks": [],
                "web_search_queries": ["Apa itu KRISNA?"],
                "grounding_supports": [object()],
            },
        )()
        candidate = type("Candidate", (), {"grounding_metadata": grounding_metadata})()
        response = type(
            "Response",
            (),
            {
                "text": "KRISNA adalah aplikasi perencanaan dan penganggaran.",
                "candidates": [candidate],
            },
        )()

        result = self.rag._parse_response(response)

        self.assertFalse(result["found"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["error"], "missing_grounding_metadata")
        self.assertNotIn("lebih spesifik", result["message"])

    def test_parse_response_preserves_grounded_no_answer_text(self) -> None:
        self.rag.file_search_service = type(
            "DummyFileSearchService",
            (),
            {
                "extract_grounding_chunks": staticmethod(
                    lambda response: [{"source_file": "Manual.pdf", "text": "Tidak ada jawaban di bagian ini."}]
                ),
                "extract_used_files": staticmethod(lambda response: ["Manual.pdf"]),
                "has_grounding": staticmethod(lambda response: True),
            },
        )()
        response = type("Response", (), {"text": "Jawaban belum tersedia di dokumen KRISNA."})()

        result = self.rag._parse_response(response)

        self.assertTrue(result["found"])
        self.assertEqual(result["answer"], "Jawaban belum tersedia di dokumen KRISNA.")
        self.assertEqual(result["error"], "")
        self.assertEqual(result["used_files"], ["Manual.pdf"])
        self.assertEqual(
            result["grounding_chunks"],
            [{"source_file": "Manual.pdf", "text": "Tidak ada jawaban di bagian ini."}],
        )
        self.assertEqual(result["message"], "")

    def test_generate_answer_reports_missing_model_name(self) -> None:
        self.override_setting("model_name", "")

        result = self.rag._generate_answer("Apa itu KRISNA?")

        self.assertFalse(result["found"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["error"], "model_not_found")
        self.assertIn("MODEL_NAME", result["message"])

    def test_generate_answer_treats_504_deadline_as_provider_unavailable(self) -> None:
        self.override_setting("model_name", "configured-model")

        def fake_call(prompt: str, model_name: str, store_names: list[str]):
            del prompt, model_name, store_names
            raise Exception(
                "504 DEADLINE_EXCEEDED. {'error': {'code': 504, "
                "'message': 'Deadline expired before operation could complete.'}}"
            )

        self.rag._get_store_names = lambda: ["fileSearchStores/test"]
        self.rag._call_model = fake_call

        result = self.rag._generate_answer("Saya tidak bisa menambah kegiatan KRISNA Renja")

        self.assertFalse(result["found"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["error"], "provider_unavailable")
        self.assertNotIn("DEADLINE_EXCEEDED", result["message"])

    def test_generate_answer_treats_server_error_503_as_provider_unavailable(self) -> None:
        self.override_setting("model_name", "configured-model")
        calls = {"count": 0}

        def fake_call(prompt: str, model_name: str, store_names: list[str]):
            del prompt, model_name, store_names
            calls["count"] += 1
            raise ServerError(
                503,
                {
                    "error": {
                        "code": 503,
                        "message": "This model is currently experiencing high demand.",
                        "status": "UNAVAILABLE",
                    }
                },
            )

        self.rag._get_store_names = lambda: ["fileSearchStores/test"]
        self.rag._call_model = fake_call

        result = self.rag._generate_answer("Apa itu KRISNA?")

        self.assertFalse(result["found"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["error"], "provider_unavailable")
        self.assertEqual(calls["count"], self.rag._retry_attempts + 1)

    def test_generate_answer_retries_503_then_succeeds(self) -> None:
        self.override_setting("model_name", "configured-model")
        calls = {"count": 0}
        self.rag.file_search_service = type(
            "DummyFileSearchService",
            (),
            {
                "extract_grounding_chunks": staticmethod(
                    lambda response: [{"source_file": "Manual.pdf", "text": "Langkah ada di menu Rincian Output."}]
                ),
                "extract_used_files": staticmethod(lambda response: ["Manual.pdf"]),
                "has_grounding": staticmethod(lambda response: True),
            },
        )()

        def fake_call(prompt: str, model_name: str, store_names: list[str]):
            del prompt, model_name, store_names
            calls["count"] += 1
            if calls["count"] == 1:
                raise ServerError(
                    503,
                    {
                        "error": {
                            "code": 503,
                            "message": "This model is currently experiencing high demand.",
                            "status": "UNAVAILABLE",
                        }
                    },
                )
            return type("Response", (), {"text": "Buka menu Rincian Output lalu klik Tambah Data."})()

        self.rag._get_store_names = lambda: ["fileSearchStores/test"]
        self.rag._call_model = fake_call

        result = self.rag._generate_answer("Bagaimana cara menambah rincian output?")

        self.assertTrue(result["found"])
        self.assertEqual(result["answer"], "Buka menu Rincian Output lalu klik Tambah Data.")
        self.assertEqual(result["used_files"], ["Manual.pdf"])
        self.assertEqual(calls["count"], 2)

    def test_generate_answer_redacts_unexpected_error_message(self) -> None:
        self.override_setting("model_name", "configured-model")

        def fake_call(prompt: str, model_name: str, store_names: list[str]):
            del prompt, model_name, store_names
            raise RuntimeError("internal provider payload with sensitive detail")

        self.rag._get_store_names = lambda: ["fileSearchStores/test"]
        self.rag._call_model = fake_call

        result = self.rag._generate_answer("Apa itu KRISNA?")

        self.assertFalse(result["found"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["error"], "unexpected_generation_error")
        self.assertNotIn("sensitive detail", result["message"])

    def test_looks_like_unanswered_answer_matches_extended_indonesian_patterns(self) -> None:
        self.assertTrue(self.rag._looks_like_unanswered_answer("Dokumen tidak memuat informasi tersebut."))
        self.assertTrue(self.rag._looks_like_unanswered_answer("Hal ini tidak dijelaskan dalam dokumen."))
        self.assertTrue(self.rag._looks_like_unanswered_answer("Jawaban ini tidak dapat saya temukan."))

    def test_looks_like_related_object_substitution_matches_self_contradicting_answer(self) -> None:
        self.assertTrue(
            self.rag._looks_like_related_object_substitution(
                "Rujukan tidak secara spesifik menampilkan langkah RO, namun menjelaskan KRO."
            )
        )
        self.assertTrue(
            self.rag._looks_like_related_object_substitution(
                "Objek ini merupakan wadah untuk objek yang ditanyakan."
            )
        )
        self.assertFalse(
            self.rag._looks_like_related_object_substitution(
                "Masuk ke halaman KRO, pilih nomenklatur, lalu pada halaman Rincian Output klik Tambah Data."
            )
        )


if __name__ == "__main__":
    unittest.main()
