# ClearLegal AI

An AI-powered assistant that answers legal questions in plain English, grounded in your own uploaded documents through retrieval-augmented generation (RAG).

## Overview

ClearLegal AI lets you upload a legal document (PDF) and ask questions about it in plain language. Instead of relying purely on the model's general knowledge, the assistant retrieves the most relevant passages from your document and grounds its answer in that content. It's guardrailed with a system prompt that keeps it from giving legal advice, reminds users that answers vary by jurisdiction, and points them to a licensed attorney for anything beyond general information.

## Features

- **Landing page** introducing the product and its value proposition
- **PDF upload & indexing** — upload a legal document; the app extracts text per page, chunks it, embeds it, and stores it in a persistent vector database
- **Retrieval-augmented chat** — questions are answered using the most relevant retrieved passages from the uploaded document, not just the model's general knowledge
- **Terms & Conditions page** with a redesigned, readable layout
- **Guardrailed responses** — restricted to legal *information* (not advice), instructed to answer only from retrieved context, capped response length, plain-text (no markdown) replies
- **Health check endpoint** for monitoring/deployment

## Technology

- **Backend:** FastAPI, Jinja2 templates, served via Uvicorn
- **Document processing:** PyMuPDF (`fitz`) for page-level PDF text extraction, LangChain's `RecursiveCharacterTextSplitter` for chunking
- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Vector store:** ChromaDB (persistent, local)
- **LLM access:** OpenAI Python SDK configured against the OpenRouter API, calling `google/gemini-2.5-flash`
- **Frontend:** Vanilla HTML/CSS
- **Containerization:** Docker + docker-compose

## Requirements

| Method | Route        | Description                                                        |
|--------|--------------|----------------------------------------------------------------------|
| GET    | `/`          | Landing page                                                        |
| GET    | `/chat`      | Chat UI                                                              |
| GET    | `/terms`     | Terms & Conditions page                                             |
| GET    | `/health`    | Health check — returns service status                               |
| POST   | `/upload`    | Upload a PDF; extracts, chunks, embeds, and indexes it into the vector store |
| POST   | `/api/chat`  | Send `{ "message": "..." }`, returns `{ "reply": "..." }` grounded in retrieved context |

## How the RAG Pipeline Works

1. **Upload** — a PDF is sent to `/upload`
2. **Extract** — PyMuPDF pulls text out page by page, keeping the page number attached
3. **Chunk** — each page's text is split into ~700-character chunks (100-char overlap), each still tagged with its source page
4. **Embed** — chunks are encoded with `all-MiniLM-L6-v2`
5. **Store** — chunks, embeddings, and page metadata are persisted in a local ChromaDB collection
6. **Retrieve** — on each question, semantic candidates from ChromaDB and lexical candidates from BM25 are fused, then a cross-encoder reranks the combined candidates and returns the top 4 chunks
7. **Generate** — retrieved passages are assembled into context and passed to Gemini 2.5 Flash, along with the legal guardrails, to produce the final answer

## Configuration

Create a `.env` file in the project root:

```env
key=your_openrouter_api_key_here
```

The application reads the API key from the `key` environment variable. Do not commit `.env` or expose the key in client-side code.

## Run locally

```bash
git clone https://github.com/Mayankk1207/ClearLegalAI.git
cd ClearLegalAI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Visit `http://localhost:8000`, upload a PDF, then ask questions about it in the chat UI.

## Run with Docker Compose

Make sure Docker Desktop is running, then:

```bash
docker compose up --build
```

Open <http://localhost:8000>. Stop the service with:

```bash
docker compose down
```

Docker Compose stores the Chroma database and uploaded PDFs in named volumes:

- `chroma_data` → `/app/chroma_db`
- `uploads_data` → `/app/uploads`

These volumes keep document data available when the container is recreated. To remove the containers and stored data:

```bash
docker compose down -v
```

## API routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Landing page |
| `GET` | `/chat` | Chat interface |
| `GET` | `/terms` | Terms & Conditions page |
| `GET` | `/health` | Service health status |
| `GET` | `/favicon.ico` | Redirects to the static SVG favicon |
| `POST` | `/upload` | Upload and index a PDF document |
| `POST` | `/api/chat` | Send `{ "message": "..." }` and receive `{ "reply": "..." }` |
| `GET` | `/static/...` | Serves frontend static assets |

Example health check:

```bash
curl http://localhost:8000/health
```

## Project structure

```text
ClearLegalAI/
├── main.py                  # FastAPI app, routes, upload + chat endpoints
├── rag/
│   ├── loader.py             # PDF text extraction (page-tracked) + ingestion entrypoint
│   ├── splitter.py           # Chunking with page metadata preserved
│   ├── embedding.py          # Sentence-Transformers embedding model
│   ├── vectorstore.py        # ChromaDB client, collection, add_chunks()
│   ├── retrieval.py          # Query embedding + top-k similarity search
│   └── pipeline.py           # Orchestrates retrieve -> prompt -> generate
├── service/
│   ├── api.py                 # LLM client (OpenRouter) and ask_llm()
│   └── prompts.py              # System prompt / guardrails + build_prompt()
├── templates/
│   ├── index.html              # Landing page
│   ├── chatbot.html             # Chat UI
│   └── terms.html                # Terms & Conditions page
├── static/
│   └── style.css
├── uploads/                  # Uploaded PDFs (runtime-generated)
├── chroma_db/                # Persistent vector store (runtime-generated)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Known Issues / Roadmap

- **Page-number citations are incomplete** — page numbers are stored per chunk in ChromaDB metadata, but the retrieval pipeline currently only passes chunk text into the prompt, not the page metadata, so the model can't reliably cite them yet
- No persistent chat history across sessions
- No cleanup or multi-user isolation for uploaded PDFs / the Chroma index
- Single LLM provider (OpenRouter/Gemini) — no configurable backend yet
