# VIM RAG MVP (FastAPI + Qdrant + OpenAI)

Minimal backend to let a Custom GPT **save** knowledge (`/ingest`) and **retrieve** relevant context (`/search`) from a Qdrant vector DB.

## Environment variables
- `QDRANT_URL` – your Qdrant endpoint (e.g., https://xxxx.qdrant.io)
- `QDRANT_API_KEY`
- `COLLECTION_NAME` – default: `vim_knowledge`
- `EMBEDDING_MODEL` – default: `text-embedding-3-large`
- `VECTOR_SIZE` – default: `3072` (use 1536 if you pick `text-embedding-3-small`)
- `OPENAI_API_KEY`
- `APP_KEY` – any string; this will be required in header `X-App-Key` for both endpoints

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

## Deploy (Render example)
1. Push these files to a GitHub repo.
2. Create a new **Web Service** on Render.
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add environment variables above.
6. Deploy and copy your public URL (e.g., `https://vim-rag.onrender.com`).

## Custom GPT Action
- In the Custom GPT builder → **Actions** → **Add Action** → **Import from file**.
- Use `openapi.yaml` (replace `https://YOUR_DEPLOY_URL` with your real URL).
- Choose **API Key (Header)** auth, header name `X-App-Key`, and paste your `APP_KEY` value there.

### Usage pattern inside your Custom GPT
- To **save** something you learned in a chat:
  - You: `aprende: <resumo objetivo do que concluímos>`
  - GPT Action → POST `/ingest` with your summary.
- To **answer questions**, the GPT should first call `POST /search` with your query and use the results to ground the answer.
