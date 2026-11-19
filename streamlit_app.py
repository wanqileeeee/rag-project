from __future__ import annotations

import hashlib
from typing import Any, Sequence
from uuid import uuid4

import streamlit as st

from utility import check_password
from utils.rag import (
    CORPUS_URLS,
    OFFICIAL_PDF_DIR,
    CorpusLoadResult,
    RAGError,
    RAGResponse,
    SourceCitation,
    add_documents_to_retriever,
    answer_question,
    build_setup_guide,
    create_retriever,
    generate_chat_title,
    is_openai_configured,
    load_corpus,
    load_uploaded_document,
)


@st.cache_resource(show_spinner=False)
def get_official_index() -> CorpusLoadResult:
    return load_corpus()


def render_sources(sources: Sequence[SourceCitation]) -> None:
    if not sources:
        return

    with st.expander("Sources"):
        for source in sources:
            label = f"[{source.citation_id}]"
            if source.page is not None:
                label += f" — page {source.page}"
            if source.source.startswith(("http://", "https://")):
                st.markdown(f"{label} — [Official guidance]({source.source})")
            else:
                qualifier = (
                    " (user-provided)" if source.kind == "user_upload" else ""
                )
                st.text(f"{label} — {source.source}{qualifier}")


def index_uploaded_files(
    uploaded_files: Sequence[Any],
    chat: dict[str, Any],
) -> list[str]:
    indexed_hashes = chat.setdefault("indexed_upload_hashes", set())
    retriever = chat.get("upload_retriever")
    added_names = []

    for uploaded_file in uploaded_files:
        content = uploaded_file.getvalue()
        fingerprint = hashlib.sha256(
            uploaded_file.name.encode("utf-8") + b"\0" + content
        ).hexdigest()
        if fingerprint in indexed_hashes:
            continue

        documents = load_uploaded_document(uploaded_file.name, content)
        if retriever is None:
            retriever = create_retriever(documents)
        else:
            add_documents_to_retriever(retriever, documents)

        indexed_hashes.add(fingerprint)
        chat["upload_retriever"] = retriever
        chat.setdefault("uploaded_names", []).append(uploaded_file.name)
        added_names.append(uploaded_file.name)

    return added_names


def render_chat_response(chat: dict[str, Any], response: RAGResponse) -> None:
    chat["messages"].append(
        {
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
        }
    )


def create_chat() -> dict[str, Any]:
    return {
        "id": uuid4().hex,
        "title": "New chat",
        "messages": [],
        "upload_retriever": None,
        "indexed_upload_hashes": set(),
        "uploaded_names": [],
    }


def chat_title(text: str) -> str:
    title = " ".join(text.split())
    return title if len(title) <= 36 else f"{title[:33].rstrip()}..."


def initialize_chats() -> None:
    if "chats" not in st.session_state:
        chat = create_chat()
        chat["messages"] = st.session_state.pop("chat_messages", [])
        chat["upload_retriever"] = st.session_state.pop("upload_retriever", None)
        chat["indexed_upload_hashes"] = st.session_state.pop(
            "indexed_upload_hashes", set()
        )
        chat["uploaded_names"] = st.session_state.pop("uploaded_names", [])
        first_question = next(
            (
                message["content"]
                for message in chat["messages"]
                if message.get("role") == "user"
            ),
            "",
        )
        if first_question:
            chat["title"] = chat_title(first_question)
        st.session_state["chats"] = [chat]
        st.session_state["active_chat_id"] = chat["id"]

    chat_ids = {chat["id"] for chat in st.session_state["chats"]}
    if st.session_state.get("active_chat_id") not in chat_ids:
        st.session_state["active_chat_id"] = st.session_state["chats"][0]["id"]


st.set_page_config(
    page_title="BizStart",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="expanded",
)
if not check_password():
    st.stop()

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 860px;
        padding-top: 1.5rem;
        padding-bottom: 5rem;
    }
    [data-testid="stChatMessage"] {
        padding: 0.8rem 1rem;
        border-radius: 0.8rem;
    }
    [data-testid="stChatInput"] {
        border-radius: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

official_index = st.session_state.get("official_index")
load_error = st.session_state.get("official_load_error")
if official_index is None and load_error is None:
    if not is_openai_configured():
        load_error = (
            "Add OPENAI_API_KEY to `.env` to load and search official guidance."
        )
    else:
        try:
            with st.spinner("Loading official Singapore business guidance..."):
                official_index = get_official_index()
            st.session_state["official_index"] = official_index
            st.session_state.pop("official_load_error", None)
        except (RAGError, OSError, ValueError) as exc:
            load_error = f"Official guidance could not be loaded: {exc}"
            st.session_state["official_load_error"] = load_error

official_retriever = (
    official_index.retriever if official_index is not None else None
)
source_warnings = official_index.warnings if official_index is not None else ()

initialize_chats()
active_chat = next(
    chat
    for chat in st.session_state["chats"]
    if chat["id"] == st.session_state["active_chat_id"]
)

with st.sidebar:
    st.markdown("## ✈️ BizStart")
    if official_retriever is not None:
        st.success(
            f"Official guidance ready "
            f"({official_index.document_count} source document(s))"
        )
        if source_warnings:
            st.caption("Some sources could not be indexed; check About Us.")
    else:
        st.warning("Official guidance is unavailable")

    if st.button("＋ New chat", type="tertiary", use_container_width=True):
        chat = create_chat()
        st.session_state["chats"].insert(0, chat)
        st.session_state["active_chat_id"] = chat["id"]
        st.rerun()

    st.divider()
    st.caption("Chats")
    for chat in st.session_state["chats"]:
        if st.button(
            (
                f"**{chat['title']}**"
                if chat["id"] == st.session_state["active_chat_id"]
                else chat["title"]
            ),
            key=f"select-chat-{chat['id']}",
            type="tertiary",
            use_container_width=True,
        ):
            st.session_state["active_chat_id"] = chat["id"]
            st.rerun()

    if st.button("Reload official sources", use_container_width=True):
        get_official_index.clear()
        st.session_state.pop("official_index", None)
        st.session_state.pop("official_load_error", None)
        st.rerun()

    st.caption(
        "Chats and uploaded reference documents stay in this browser session."
    )

about_tab, qa_tab, guide_tab, methodology_tab = st.tabs(
    [
        "About Us",
        "Business Q&A",
        "Personalised guide setup",
        "Methodology",
    ]
)

with about_tab:
    st.title("About Us")
    st.write(
        "BizStart is a Singapore-focused retrieval-augmented generation (RAG) "
        "assistant for people starting or operating a small business."
    )
    st.subheader("Project scope")
    st.write(
        "The project helps users find, interpret, and organise publicly available "
        "Singapore business guidance. It is designed around business registration, "
        "ownership structures, GST, home-based businesses, online selling, and "
        "licences or permits. It does not replace formal legal, tax, or regulatory advice."
    )
    st.subheader("Objectives")
    st.markdown(
        """
- Make Singapore business guidance easier to discover and navigate.
- Answer questions using evidence retrieved from official sources and named user references.
- Produce a personalised setup checklist grounded in the user's activity, ownership,
  premises, and revenue inputs.
- Preserve source transparency through page-level or document-level citations.
- Clearly indicate when indexed evidence is unavailable or insufficient.
"""
    )
    st.subheader("Data sources")
    st.write(
        "The shared corpus combines Singapore government web guidance from ACRA, CPF, "
        "GoBusiness, URA, and IRAS; TikTok Shop Singapore guidance; Singapore Customs "
        "guidance; and locally stored, text-searchable Singapore government PDFs."
    )
    st.markdown(
        """
The current local PDF collection covers HDB/URA home-business schemes and SFA
licence/permit requirements, fees, applications, and temporary fair operations.
The app also accepts PDF and UTF-8 TXT references uploaded within an individual chat.
Uploaded references are session-owned and are used only in that chat.
"""
    )
    st.subheader("Features")
    st.markdown(
        """
- **Business Q&A:** conversational answers with short-query follow-up context,
  official evidence, user-uploaded evidence, and citations.
- **Personalised guide setup:** a generated structure comparison and licence/next-steps
  checklist for the supplied business profile.
- **Multiple chats:** separate browser-session chat histories and uploaded references.
- **Source status:** indexed-source counts, load errors, unavailable sources, and
  reload controls.
- **Documentation:** this About Us page and the accompanying Methodology page.
"""
    )
    st.info(
        "For information only—not legal or tax advice. Confirm current "
        "requirements with ACRA, GoBusiness, and IRAS."
    )

    if load_error:
        st.warning(load_error)
        if is_openai_configured() and st.button("Retry loading official guidance"):
            st.session_state.pop("official_load_error", None)
            st.rerun()

    if source_warnings:
        st.warning(
            "Some official sources could not be indexed. Answers can only use "
            "sources that were successfully indexed."
        )
        with st.expander("Official sources not indexed"):
            for warning in source_warnings:
                if warning.source.startswith(("http://", "https://")):
                    st.markdown(
                        f"- [{warning.source}]({warning.source}) — {warning.message}"
                    )
                else:
                    st.markdown(f"- `{warning.source}` — {warning.message}")

    with st.expander("Official source collection", expanded=True):
        unavailable_sources = {warning.source for warning in source_warnings}
        for url in CORPUS_URLS:
            status = " — not indexed" if url in unavailable_sources else ""
            st.markdown(f"- [{url}]({url}){status}")
        st.caption(f"Local PDFs: `{OFFICIAL_PDF_DIR}`")
        if OFFICIAL_PDF_DIR.exists():
            for pdf_path in sorted(OFFICIAL_PDF_DIR.iterdir()):
                if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
                    continue
                status = (
                    " — not indexed"
                    if str(pdf_path) in unavailable_sources
                    or pdf_path.name in unavailable_sources
                    else ""
                )
                st.text(f"- {pdf_path.name}{status}")

    st.caption(
        "Document text and retrieved excerpts are sent to OpenAI for embeddings "
        "and answer generation. Do not attach confidential or sensitive material."
    )

with qa_tab:
    chat_messages = active_chat["messages"]
    if not chat_messages:
        st.markdown("### What would you like to know?")
        st.write(
            "Ask a question about business registration, GST, business structures, "
            "home businesses, licences, or selling online."
        )
        examples = (
            "When do I need to register for GST?",
            "What is the difference between an LLP and a Pte Ltd?",
            "Can I run a food business from home?",
            "What licences might an online shop need?",
        )
        question = None
        for row_start in (0, 2):
            example_columns = st.columns(2)
            for column, example in zip(
                example_columns,
                examples[row_start : row_start + 2],
            ):
                with column:
                    with st.container(border=True):
                        st.markdown(
                            f"**{example}**"
                        )
                        if st.button(
                            "Ask this",
                            key=f"example-{row_start}-{example}",
                            use_container_width=True,
                            disabled=official_retriever is None,
                        ):
                            question = example
    else:
        question = None

    if active_chat["uploaded_names"]:
        st.caption(
            "References in this chat: "
            + ", ".join(active_chat["uploaded_names"])
        )

    for message in chat_messages:
        avatar = "✈️" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message.get("attachments"):
                st.caption("Attached: " + ", ".join(message["attachments"]))
            if message.get("sources"):
                render_sources(message["sources"])

    submission = st.chat_input(
        "Message BizStart or add a reference...",
        disabled=official_retriever is None,
        accept_file="multiple",
        file_type=["pdf", "txt"],
        key=f"chat-input-{active_chat['id']}",
    )
    submitted_files = []
    if submission is not None:
        if isinstance(submission, str):
            question = submission
        else:
            question = submission.text
            submitted_files = submission.files

    attachment_names = []
    upload_failed = False
    if submitted_files:
        try:
            with st.spinner("Reading and indexing reference documents..."):
                attachment_names = index_uploaded_files(
                    submitted_files,
                    active_chat,
                )
            if attachment_names:
                st.success(
                    f"Added {len(attachment_names)} reference document(s) to this chat."
                )
            else:
                st.info("These reference documents are already in this chat.")
        except (RAGError, OSError, ValueError) as exc:
            st.error(f"Could not add reference documents: {exc}")
            attachment_names = []
            upload_failed = True

    if question or (submitted_files and not upload_failed):
        if question:
            message_content = question
        else:
            message_content = (
                "Added reference documents."
                if attachment_names
                else "These reference documents are already in this chat."
            )
        active_chat["messages"].append(
            {
                "role": "user",
                "content": message_content,
                "attachments": (
                    [uploaded_file.name for uploaded_file in submitted_files]
                    if not upload_failed
                    else []
                ),
            }
        )
        if active_chat["title"] == "New chat":
            if question:
                try:
                    active_chat["title"] = generate_chat_title(question)
                except RAGError as exc:
                    st.warning(f"Could not generate a chat title: {exc}")
            elif submitted_files:
                active_chat["title"] = chat_title(
                    f"References: {submitted_files[0].name}"
                )
        if not question:
            st.rerun()

        history = list(active_chat["messages"][:-1])
        try:
            with st.spinner("Searching official guidance..."):
                response = answer_question(
                    question,
                    official_retriever=official_retriever,
                    upload_retriever=active_chat["upload_retriever"],
                    history=history,
                )
            render_chat_response(active_chat, response)
        except (RAGError, OSError, ValueError) as exc:
            render_chat_response(
                active_chat,
                RAGResponse(
                    answer=f"Could not answer this question: {exc}",
                    sources=[],
                )
            )
        st.rerun()

    if official_retriever is None:
        st.info("The chat will be ready when official guidance has loaded.")

with guide_tab:
    st.subheader("Build your business setup guide")
    st.write(
        "Tell us a little about your plan. BizStart will retrieve relevant "
        "Singapore guidance and organise it into suggested next steps."
    )
    st.info(
        "For information only—not legal or tax advice. Confirm current "
        "requirements with ACRA, GoBusiness, and IRAS."
    )

    with st.form("business_setup_form", border=True):
        activity = st.text_input(
            "What will your business do?",
            placeholder="e.g. home-based bakery, TikTok shop, freelance design",
        )
        owner_count = st.selectbox("Number of owners", ["1", "2", "3 or more"])
        premises = st.selectbox(
            "Where will it operate?",
            ["Home-based", "Commercial premises", "Online only", "Not decided"],
        )
        revenue = st.selectbox(
            "Expected annual revenue",
            [
                "Below S$100,000",
                "S$100,000–S$499,999",
                "S$500,000–S$999,999",
                "S$1,000,000 or more",
                "Not sure",
            ],
        )
        submitted = st.form_submit_button(
            "Create my setup guide",
            type="primary",
            use_container_width=True,
            disabled=official_retriever is None,
        )

    if submitted:
        active_chat.pop("setup_result", None)
        if not activity.strip():
            st.error("Describe your business activity to generate a relevant guide.")
        else:
            try:
                with st.spinner("Finding guidance for your business..."):
                    active_chat["setup_result"] = build_setup_guide(
                        activity=activity.strip(),
                        owner_count=owner_count,
                        premises=premises,
                        revenue=revenue,
                        official_retriever=official_retriever,
                        upload_retriever=active_chat["upload_retriever"],
                    )
            except (RAGError, OSError, ValueError) as exc:
                st.error(f"Could not generate the setup guide: {exc}")

    setup_result = active_chat.get("setup_result")
    if setup_result:
        st.markdown(setup_result.answer)
        render_sources(setup_result.sources)

with methodology_tab:
    st.title("Methodology")
    st.write(
        "The following sections describe the implementation data flow and the "
        "processing sequence for each application use case."
    )

    st.subheader("Overall implementation")
    st.markdown(
        """
1. At startup, BizStart checks for an OpenAI API key.
2. Official web pages are fetched by URL and local PDF files are extracted page by page.
3. Documents are divided into overlapping text chunks of 1,000 characters with
   200-character overlap.
4. OpenAI text-embedding-3-small creates vector embeddings, and an in-memory
   Chroma collection stores the shared official corpus.
5. A question or setup profile is embedded through the retriever and the four most
   relevant chunks are requested.
6. Retrieved official and chat-uploaded passages are assembled as labelled evidence.
7. GPT-4o-mini generates a grounded answer or setup guide using only that evidence.
8. The response is returned with source markers, source names, and PDF page numbers
   where available.
"""
    )
    st.caption(
        "The OpenAI key is required for embeddings and answer generation. "
        "Source text and retrieved excerpts are sent to OpenAI."
    )

    st.subheader("Use case 1: Business Q&A")
    st.write(
        "A user asks a question, optionally in a chat containing uploaded references."
    )
    st.graphviz_chart(
        """
digraph qa {
    rankdir=LR;
    node [shape=box, style="rounded", fontname="Helvetica"];
    input [label="User question"];
    history [label="Last 6 chat messages"];
    retrieve [label="Retrieve up to 4 chunks\nfrom official + chat indexes"];
    evidence [label="Deduplicate citations\nand build evidence"];
    generate [label="GPT-4o-mini answer"];
    output [label="Answer + source markers"];
    input -> retrieve;
    history -> retrieve [label="context for short questions"];
    retrieve -> evidence -> generate -> output;
}
"""
    )
    st.markdown(
        "- Short follow-up questions reuse the previous user question in the retrieval query.\n"
        "- Official and uploaded retrievers are queried independently, then combined.\n"
        "- The prompt requires factual claims to be cited and says to report insufficient evidence."
    )

    st.subheader("Use case 2: Personalised setup guide")
    st.write(
        "The setup form combines the user's activity, owner count, premises, and revenue."
    )
    st.graphviz_chart(
        """
digraph guide {
    rankdir=LR;
    node [shape=box, style="rounded", fontname="Helvetica"];
    form [label="Business profile form"];
    validate [shape=diamond, label="Activity entered?"];
    query [label="Build setup retrieval query"];
    retrieve [label="Retrieve official + chat evidence"];
    generate [label="Generate grounded guide"];
    sections [label="Structure, comparison,\nlicences + next steps"];
    form -> validate;
    validate -> query [label="yes"];
    validate -> form [label="no"];
    query -> retrieve -> generate -> sections;
}
"""
    )

    st.subheader("Use case 3: Add reference documents")
    st.write(
        "A PDF or UTF-8 TXT file uploaded to a chat creates chat-only retrieval evidence."
    )
    st.graphviz_chart(
        """
digraph upload {
    rankdir=LR;
    node [shape=box, style="rounded", fontname="Helvetica"];
    file [label="User uploads PDF or TXT"];
    type [shape=diamond, label="Supported type?"];
    extract [label="Extract text/page text"];
    hash [shape=diamond, label="Already indexed?"];
    chunk [label="Split + embed document"];
    index [label="Add to chat retriever"];
    reuse [label="Skip duplicate"];
    file -> type;
    type -> extract [label="yes"];
    type -> file [label="no"];
    extract -> hash;
    hash -> reuse [label="yes"];
    hash -> chunk [label="no"];
    chunk -> index;
}
"""
    )

    st.subheader("Use case 4: Load or reload official sources")
    st.write(
        "Official sources are loaded on first use or explicitly reloaded from the sidebar."
    )
    st.graphviz_chart(
        """
digraph corpus {
    rankdir=LR;
    node [shape=box, style="rounded", fontname="Helvetica"];
    trigger [label="App startup or reload"];
    fetch [label="Fetch web pages"];
    pdf [label="Read local PDF pages"];
    loadable [shape=diamond, label="Readable text?"];
    warning [label="Record source warning"];
    embed [label="Chunk + embed"];
    retriever [label="Create shared retriever"];
    trigger -> fetch;
    trigger -> pdf;
    fetch -> loadable;
    pdf -> loadable;
    loadable -> warning [label="no"];
    loadable -> embed [label="yes"];
    embed -> retriever;
}
"""
    )

    st.subheader("Grounding and failure handling")
    st.markdown(
        """
- Retrieval failures and missing source evidence return an explicit insufficient-evidence
  or unavailable-guidance message rather than inventing an answer.
- Fetch or parsing failures are retained as source warnings and do not automatically
  discard the successfully loaded part of the corpus.
- A source marker is de-duplicated by source name and PDF page number.
- OpenAI and document-load errors are translated into visible RAG error responses.
- Answers are informational, and the app directs users to confirm requirements with
  ACRA, GoBusiness, and IRAS or the relevant regulator.
"""
    )
