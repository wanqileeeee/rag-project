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
    SourceCitation,
    _collect_evidence,
    _retrieval_query,
    answer_question,
    build_setup_guide,
    create_retriever,
    generate_chat_title,
    load_corpus_documents,
    load_official_pdf_documents,
    load_uploaded_document,
    link_citations,
    source_label,
)


class StaticRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
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


class SourceCitationTests(unittest.TestCase):
    def test_web_citation_uses_page_title_and_links_to_url(self):
        document = Document(
            page_content="GST information",
            metadata={
                "source": "https://iras.gov.sg/gst",
                "title": "GST Registration | IRAS",
            },
        )

        self.assertEqual(
            source_label(document),
            ("GST Registration", "https://iras.gov.sg/gst"),
        )

    def test_pdf_citation_uses_title_and_existing_one_based_page_number(self):
        document = Document(
            page_content="Home business guidance",
            metadata={
                "source": "Home-Based Business Scheme.pdf",
                "title": "Home Business Scheme",
                "page": 2,
            },
        )

        self.assertEqual(
            source_label(document),
            ("Home Business Scheme, p.2", None),
        )

    def test_answer_citations_become_web_links_or_pdf_labels(self):
        sources = [
            SourceCitation(
                citation_id="S1",
                source="https://iras.gov.sg/gst",
                kind="official",
                label="GST Registration",
                url="https://iras.gov.sg/gst",
            ),
            SourceCitation(
                citation_id="S2",
                source="home-business.pdf",
                kind="official",
                page=2,
                label="Home Business Scheme, p.2",
            ),
        ]

        answer = link_citations("See [S1] and [S2].", sources)

        self.assertEqual(
            answer,
            "See ([GST Registration](<https://iras.gov.sg/gst>)) and "
            "(*Home Business Scheme, p.2*).",
        )
        self.assertEqual(link_citations("Unsupported [S9].", sources), "Unsupported.")

    def test_citation_variants_are_merged_and_dollars_escaped(self):
        sources = [
            SourceCitation(
                citation_id="S1",
                source="https://iras.gov.sg/gst",
                kind="official",
                label="GST Registration",
                url="https://iras.gov.sg/gst",
            ),
            SourceCitation(
                citation_id="S2",
                source="home-business.pdf",
                kind="official",
                label="Home Business Scheme, p.2",
            ),
        ]
        link = "[GST Registration](<https://iras.gov.sg/gst>)"

        self.assertEqual(
            link_citations("Over S$1 million [S1 | official | IRAS].", sources),
            rf"Over S\$1 million ({link}).",
        )
        self.assertEqual(
            link_citations("Rules apply ([S1], [S2], [S1]).", sources),
            f"Rules apply ({link}; *Home Business Scheme, p.2*).",
        )
        self.assertEqual(
            link_citations("Fee [S2](https://sfa.gov.sg/x).", sources),
            "Fee (*Home Business Scheme, p.2*).",
        )

    def test_bare_agency_web_title_uses_page_slug(self):
        document = Document(
            page_content="GST information",
            metadata={
                "source": "https://www.iras.gov.sg/taxes/do-i-need-to-register-for-gst",
                "title": "IRAS",
            },
        )

        self.assertEqual(
            source_label(document)[0],
            "IRAS – Do i need to register for gst",
        )


class OfficialPdfTests(unittest.TestCase):
    def test_removed_web_sources_are_not_in_the_official_url_list(self):
        removed_urls = {
            "https://www.sfa.gov.sg/food-retail/businesses-that-currently-do-not-need-licence-permit/home-based-food-businesses/requirements-for-home-based-food-businesses",
            "https://www.sfa.gov.sg/food-retail/businesses-that-currently-do-not-need-licence-permit/home-based-food-businesses/about-hdb-ura-home-based-business-scheme",
            "https://www.sfa.gov.sg/food-retail/licence-permit/businesses-that-need-licence-permit-for-food-retail",
            "https://www.hdb.gov.sg/managing-my-home/home-ownership/home-business/homebased-business-scheme",
            "https://www.gobusiness.gov.sg/",
            "https://licensing.gobusiness.gov.sg/e-adviser/home-based-business/home-based-business-scheme",
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
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("utils.rag.ChatOpenAI")
    def test_chat_retrieval_uses_question_and_distinct_subquestions(self, chat_model):
        chat_model.return_value.invoke.return_value = SimpleNamespace(
            content=(
                "- Singapore business structure\n"
                "2. Singapore business licences\n"
                "Singapore business structure\n"
            )
        )
        document = Document(
            page_content="Registration guidance",
            metadata={"source": "https://acra.gov.sg/register", "kind": "official"},
        )
        retriever = StaticRetriever([document])

        answer_question(
            "How do I start a business?",
            official_retriever=retriever,
        )

        self.assertEqual(
            retriever.queries,
            [
                "How do I start a business?",
                "Singapore business structure",
                "Singapore business licences",
            ],
        )
        answer_system_prompt = chat_model.return_value.invoke.call_args_list[1].args[
            0
        ][0].content
        self.assertIn("Only state numbers, rates, thresholds and dates", answer_system_prompt)
        self.assertIn("Never take a threshold figure from the user's inputs", answer_system_prompt)
        self.assertIn("correct it explicitly at the start", answer_system_prompt)
        self.assertIn("Can I sell handmade candles from my HDB flat?", answer_system_prompt)
        self.assertIn("Never use facts from the examples", answer_system_prompt)

    @patch("utils.rag.requests.get")
    def test_unavailable_official_pages_are_reported_but_do_not_block_indexing(
        self, get
    ):
        get.side_effect = [
            FakeResponse(202, b""),
            requests.HTTPError("HTTP 403"),
            FakeResponse(
                200,
                b"<html><head><title>Business Registration | ACRA</title></head>"
                b"<body>Official guidance</body></html>",
            ),
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
        self.assertEqual(
            documents[0].metadata["title"],
            "Business Registration | ACRA",
        )
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

    def test_evidence_from_multiple_queries_deduplicates_repeated_chunks(self):
        document = Document(
            page_content="GST guidance excerpt",
            metadata={"source": "https://iras.gov.sg/gst", "kind": "official"},
        )
        retriever = StaticRetriever([document])

        context, sources = _collect_evidence(
            ["GST", "registration"],
            retriever,
            None,
        )

        self.assertEqual(retriever.queries, ["GST", "registration"])
        self.assertEqual(context.count("GST guidance excerpt"), 1)
        self.assertEqual(len(sources), 1)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("utils.rag.Chroma.from_documents")
    @patch("utils.rag._split_documents")
    def test_retrievers_use_mmr_with_expanded_candidate_pool(
        self,
        split_documents,
        from_documents,
    ):
        split_documents.return_value = [
            Document(page_content="Registration guidance")
        ]
        vector_store = from_documents.return_value

        create_retriever([Document(page_content="Registration guidance")])

        vector_store.as_retriever.assert_called_once_with(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 20},
        )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("utils.rag.ChatOpenAI")
    def test_setup_guide_retrieves_each_topic_separately_and_limits_numbers(
        self,
        chat_model,
    ):
        chat_model.return_value.invoke.return_value = SimpleNamespace(
            content="## Suggested structure\nGrounded guidance."
        )
        retriever = StaticRetriever(
            [
                Document(
                    page_content="Official business setup evidence",
                    metadata={
                        "source": "https://example.gov.sg/business",
                        "kind": "official",
                    },
                )
            ]
        )

        build_setup_guide(
            activity="home bakery",
            owner_count="1",
            premises="Home-based",
            revenue="Below S$100,000",
            official_retriever=retriever,
        )

        self.assertEqual(len(retriever.queries), 6)
        self.assertTrue(any("business structure" in query for query in retriever.queries))
        self.assertTrue(any("ACRA business registration" in query for query in retriever.queries))
        self.assertTrue(any("licences and permits" in query for query in retriever.queries))
        self.assertTrue(any("GST registration threshold" in query for query in retriever.queries))
        self.assertTrue(any("CPF MediSave" in query for query in retriever.queries))
        self.assertTrue(any("home-based business scheme" in query for query in retriever.queries))
        setup_system_prompt = chat_model.return_value.invoke.call_args.args[0][0].content
        self.assertIn("Only state numbers, rates, thresholds and dates", setup_system_prompt)
        self.assertIn(
            "Never take a threshold figure from the user's inputs", setup_system_prompt
        )
        self.assertIn("Cite only with bare markers", setup_system_prompt)
        self.assertNotIn("Can I sell handmade candles from my HDB flat?", setup_system_prompt)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("utils.rag.ChatOpenAI")
    def test_answer_uses_retrieved_sources_and_returns_citations(self, chat_model):
        chat_model.return_value.invoke.side_effect = [
            SimpleNamespace(content="Singapore business registration"),
            SimpleNamespace(content="Register based on the official guidance [S1]."),
        ]
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

        self.assertIn("[https://acra.gov.sg/register]", response.answer)
        self.assertNotIn("[S1]", response.answer)
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
