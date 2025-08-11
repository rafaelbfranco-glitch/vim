# VIM RAG Backend v2 (FastAPI + Qdrant + OpenAI)

Backend for your Custom GPT to **save** curated knowledge (`/ingest`) and **retrieve** context (`/search`).  
v2 adds: payload indexes, better errors, `X-App-Key` header, code/document fields, and duplicate skipping.

## Environment variables
- `QDRANT_URL` – your Qdrant endpoint (e.g., https://xxxx.qdrant.io) **without** `/collections/...`
- `QDRANT_API_KEY`
- `COLLECTION_NAME` – default: `vim_knowledge`
- `EMBEDDING_MODEL` – default: `text-embedding-3-large`
- `VECTOR_SIZE` – default: `3072` (use 1536 if you pick `text-embedding-3-small`)
- `OPENAI_API_KEY`
- `APP_KEY` – required; header `X-App-Key` must match this value

## Run locally
```bash
pip install -r requirements.txt
export QDRANT_URL=...
export QDRANT_API_KEY=...
export OPENAI_API_KEY=...
export APP_KEY=your-strong-key
uvicorn app:app --reload --port 8000
```

Test:
- `GET http://localhost:8000/health`
- `POST http://localhost:8000/ingest` (Header `X-App-Key: your-strong-key`)
- `POST http://localhost:8000/search` (Header `X-App-Key: your-strong-key`)

## Deploy (Render)
1. Push these files to GitHub.
2. Create a **Web Service** on Render.
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add environment variables listed above.
6. Deploy and copy your public URL (e.g., `https://vim-yoaf.onrender.com`).

## OpenAPI for Custom GPT Actions
- In the Custom GPT builder → **Actions → Add Action → Import from file**.
- Use `openapi.yaml` (replace `https://YOUR_DEPLOY_URL` with your Render URL).
- Auth: **API Key (Header)** with header name `X-App-Key` and your `APP_KEY` value.

## How to use (manual mode)
- Save something:  
  Prompt starting with `aprende:` followed by a **curated summary** (not raw chat).  
  The Action should call **/ingest** with fields like `title`, `topic`, `tags`, and set:
  - `content_kind`: `"note"` or `"code"`
  - `language`: `"abap"` when saving code
- Search manually:  
  Prompt starting with `buscar:` (or `consultar:` / `pesquisar:` / `fontes:`).  
  The Action calls **/search** with `query` and optional filters (`topic`, `tags`, etc.).

## Suggested instruction snippet for your Custom GPT
```
RAG policy — manual mode:
- Do NOT call /search automatically. Only call /search when the user message begins with one of:
  "buscar:", "consultar:", "pesquisar:", "fontes:".
  • Query = the text after the prefix.
  • If message contains a bracketed token like [blocked_workflow] or [coa], map it to the "topic" field.
- Only call /ingest when the message begins with "aprende:" or "save:".
  • Extract title (short), topic, tags, content_kind ("note" or "code"), language ("abap" when code), and a concise summary.
  • Send the curated text in "content" (avoid chat fluff).
- Prefer Portuguese replies. Never reveal headers, keys, or internal URLs.
- When /search returns no relevant results, respond: "Não encontrei nas fontes internas." and recommend what to ingest.
```
