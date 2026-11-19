# BizStart

BizStart is a Singapore-focused RAG assistant for people starting a business.
It retrieves guidance from configured official sources, cites retrieved passages,
and can also use PDF or TXT documents uploaded for the current session.

## Features

- **Business Q&A:** Ask follow-up questions about registration, GST, business
  structures, home businesses, licences, and online selling.
- **Personalised setup guide:** Provide a business activity, owner count,
  premises type, and estimated annual revenue to receive a cited structure
  comparison and a next-steps checklist.
- **Chat history and references:** Start separate chats and attach PDF or
  UTF-8 TXT documents from the message box. References are kept with that chat
  for the browser session and labelled as user-provided. Their extracted text
  and retrieved excerpts are sent to OpenAI for embeddings and answer
  generation; do not attach confidential or sensitive information.

## Run locally

1. Install [uv](https://docs.astral.sh/uv/).
2. Create a `.env` file in the project root and set your OpenAI API key:

   ```text
   OPENAI_API_KEY=your-api-key
   ```

3. Set a password in `.streamlit/secrets.toml` (create the file if needed):

   ```toml
   APP_PASSWORD = "choose-a-long-unique-password"
   ```

   Keep this file private; do not commit it. For a hosted deployment, set
   `APP_PASSWORD` in the app's secrets configuration instead.

4. Sync dependencies and start the app:

   ```bash
   uv sync
   uv run streamlit run streamlit_app.py
   ```

5. On first startup, BizStart automatically fetches and indexes the official
   guidance. This requires an internet connection and can take a short while;
   the index is cached for the lifetime of the app process.

To add official PDF sources, place text-searchable `.pdf` files in
`data/official_pdfs/` at the project root (create the folder if needed). In the
running app, choose **Reload official sources** in the sidebar to index newly
added or changed files. Each PDF page is indexed and cited separately.

The official corpus is fetched from the URLs listed in `utils/rag.py` and from
the local PDF folder, then embedded using OpenAI. Answers are generated only
from retrieved passages; when the evidence is insufficient, BizStart is
instructed to say so. Sources that are blocked, unreadable, or return no text
are skipped and shown in the app, so check that notice for source gaps before
relying on an answer. This service is for information only, not legal or tax
advice. Confirm current requirements with ACRA, GoBusiness, and IRAS.
