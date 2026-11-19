import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document
import requests

from utils.rag import (
    CORPUS_URLS,
    _collect_evidence,
    _retrieval_query,
    answer_question,
    generate_chat_title,
    load_corpus_documents,
    load_official_pdf_documents,
    load_uploaded_document,
)


class StaticRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, query):
        return self.documents


class FakeResponse:
    def __init__(self, status_code, content):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class UploadedDocumentTests(unittest.TestCase):
    def test_loads_utf8_text_with_user_source_metadata(self):
        documents = load_uploaded_document("notes.txt", b"Business registration notes")

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "Business registration notes")
        self.assertEqual(documents[0].metadata["source"], "Upload: notes.txt")
        self.assertEqual(documents[0].metadata["kind"], "user_upload")

    def test_rejects_unsupported_upload_types(self):
        with self.assertRaisesRegex(ValueError, "PDF or TXT"):
            load_uploaded_document("notes.docx", b"not supported")


class OfficialPdfTests(unittest.TestCase):
    def test_removed_web_sources_are_not_in_the_official_url_list(self):
        removed_urls = {
            "https://www.sfa.gov.sg/food-retail/businesses-that-currently-do-not-need-licence-permit/home-based-food-businesses/requirements-for-home-based-food-businesses",
            "https://www.sfa.gov.sg/food-retail/businesses-that-currently-do-not-need-licence-permit/home-based-food-businesses/about-hdb-ura-home-based-business-scheme",
            "https://www.sfa.gov.sg/food-retail/licence-permit/businesses-that-need-licence-permit-for-food-retail",
            "https://www.hdb.gov.sg/managing-my-home/home-ownership/home-business/homebased-business-scheme",
            "https://www.gobusiness.gov.sg/",
        }

        self.assertTrue(removed_urls.isdisjoint(CORPUS_URLS))

    @patch("utils.rag.load_uploaded_document")
    def test_corpus_includes_local_pdf_as_official_document(self, load_document):
        load_document.return_value = [
            Document(
                page_content="SFA home food business guidance",
                metadata={
                    "source": "Upload:home-food.pdf",
                    "kind": "user_upload",
                    "page": 1,
                },
            )
        ]

        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "home-food.pdf"
            pdf_path.write_bytes(b"PDF content")

            documents, warnings = load_corpus_documents(
                urls=(),
                pdf_dir=Path(temp_dir),
            )

        self.assertEqual(warnings, [])
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata["source"], "home-food.pdf")
        self.assertEqual(documents[0].metadata["kind"], "official")
        self.assertEqual(documents[0].metadata["page"], 1)

    @patch("utils.rag.load_uploaded_document", side_effect=ValueError("bad PDF"))
    def test_reports_unreadable_local_pdf_without_blocking_other_sources(
        self, load_document
    ):
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "broken.pdf"
            pdf_path.write_bytes(b"not a PDF")

            documents, warnings = load_official_pdf_documents(Path(temp_dir))

        self.assertEqual(documents, [])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].source, "broken.pdf")
        self.assertIn("bad PDF", warnings[0].message)


class RetrievalTests(unittest.TestCase):
    @patch("utils.rag.requests.get")
    def test_unavailable_official_pages_are_reported_but_do_not_block_indexing(
        self, get
    ):
        get.side_effect = [
            FakeResponse(202, b""),
            requests.HTTPError("HTTP 403"),
            FakeResponse(200, b"<html><body>Official guidance</body></html>"),
        ]

        with TemporaryDirectory() as temp_dir:
            documents, warnings = load_corpus_documents(
                [
                    "https://sfa.gov.sg/blocked",
                    "https://hdb.gov.sg/blocked",
                    "https://acra.gov.sg/available",
                ],
                pdf_dir=Path(temp_dir),
            )

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "Official guidance")
        self.assertEqual(len(warnings), 2)
        self.assertEqual(warnings[0].source, "https://sfa.gov.sg/blocked")
        self.assertIn("HTTP 202", warnings[0].message)

    def test_follow_up_reuses_the_previous_user_question_for_retrieval(self):
        query = _retrieval_query(
            "And GST?",
            [{"role": "user", "content": "I am opening a home bakery"}],
        )

        self.assertIn("home bakery", query)
        self.assertIn("And GST?", query)

    def test_evidence_deduplicates_citations_by_source_and_page(self):
        documents = [
            Document(
                page_content="GST guidance excerpt",
                metadata={"source": "https://iras.gov.sg/gst", "kind": "official"},
            ),
            Document(
                page_content="Another excerpt from this page",
                metadata={"source": "https://iras.gov.sg/gst", "kind": "official"},
            ),
        ]
        context, sources = _collect_evidence(
            "GST",
            StaticRetriever(documents),
            None,
        )

        self.assertIn("[S1 | official | https://iras.gov.sg/gst]", context)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].citation_id, "S1")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("utils.rag.ChatOpenAI")
    def test_answer_uses_retrieved_sources_and_returns_citations(self, chat_model):
        chat_model.return_value.invoke.return_value = SimpleNamespace(
            content="Register based on the official guidance [S1]."
        )
        retriever = StaticRetriever(
            [
                Document(
                    page_content="Registration guidance",
                    metadata={
                        "source": "https://acra.gov.sg/register",
                        "kind": "official",
                    },
                )
            ]
        )

        response = answer_question(
            "How do I register?",
            official_retriever=retriever,
        )

        self.assertIn("[S1]", response.answer)
        self.assertEqual(response.sources[0].source, "https://acra.gov.sg/register")


class ChatTitleTests(unittest.TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("utils.rag.ChatOpenAI")
    def test_generates_clean_concise_title(self, chat_model):
        chat_model.return_value.invoke.return_value = SimpleNamespace(
            content='  "Registering a Home Bakery." \n'
        )

        title = generate_chat_title("How do I register a home bakery?")

        self.assertEqual(title, "Registering a Home Bakery")


if __name__ == "__main__":
    unittest.main()
