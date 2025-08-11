import os
import uuid
import hashlib
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    PayloadSchemaType,
)

# OpenAI embeddings
from openai import OpenAI

# ----------- Environment -----------
QDRANT_URL = os.environ.get("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip()
COLLECTION = os.environ.get("COLLECTION_NAME", "vim_knowledge").strip()

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large").strip()
VECTOR_SIZE = int(os.environ.get("VECTOR_SIZE", "3072"))
APP_KEY = os.environ.get("APP_KEY", "").strip()  # required by header X-App-Key

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set as environment variables.")

# Clients
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

# Ensure collection exists
def ensure_collection():
    existing = qdrant.get_collections().collections
    names = [c.name for c in existing]
    if COLLECTION not in names:
        qdrant.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

def ensure_payload_indexes():
    # Fields we may filter on
    fields = [
        "topic", "tags", "country", "sap_release", "vim_release",
        "content_kind", "language", "title", "hash", "customer", "project"
    ]
    for f in fields:
        try:
            qdrant.create_payload_index(
                collection_name=COLLECTION,
                field_name=f,
                field_schema=PayloadSchemaType.KEYWORD,  # string/keyword; list is fine for KEYWORD
            )
        except Exception:
            # Already exists or server unavailable -> ignore
            pass

ensure_collection()
ensure_payload_indexes()

# ----------- FastAPI -----------
app = FastAPI(title="VIM RAG Backend v2", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------- Models -----------
class IngestItem(BaseModel):
    title: Optional[str] = None                # short title to help search results
    topic: Optional[str] = None                # e.g., blocked_workflow, coa, drc
    content: str                               # main text to embed (summary or full)
    summary: Optional[str] = None              # optional short summary; stored for display
    content_kind: Optional[str] = "note"       # note | code | doc
    language: Optional[str] = None             # e.g., abap, python, pt-BR
    source: Optional[str] = None               # link/id of origin
    tags: Optional[List[str]] = None
    country: Optional[str] = None
    sap_release: Optional[str] = None
    vim_release: Optional[str] = None
    customer: Optional[str] = None
    project: Optional[str] = None
    created_at: Optional[str] = None           # ISO8601; default now if not provided
    dedup: Optional[bool] = True               # skip insert if same content hash already exists

class SearchQuery(BaseModel):
    query: str
    k: int = 6
    min_score: Optional[float] = 0.0
    topic: Optional[str] = None
    country: Optional[str] = None
    sap_release: Optional[str] = None
    vim_release: Optional[str] = None
    tags: Optional[List[str]] = None
    content_kind: Optional[str] = None
    language: Optional[str] = None
    customer: Optional[str] = None
    project: Optional[str] = None

# ----------- Helpers -----------
def require_app_key(x_app_key: Optional[str] = Header(default=None, alias="X-App-Key")):
    """Simple header-based auth for Actions & Postman."""
    if not APP_KEY:
        raise HTTPException(status_code=500, detail="Server APP_KEY not configured")
    if not x_app_key or x_app_key != APP_KEY:
        raise HTTPException(status_code=401, detail="Invalid X-App-Key")

def embed_text(text: str) -> List[float]:
    try:
        resp = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return resp.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EmbeddingError: {type(e).__name__}: {str(e)[:300]}")

def chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    """Very simple chunker by characters; safe for MVP."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def to_filter(payload: SearchQuery) -> Optional[Filter]:
    must = []
    def mv(key, val):
        nonlocal must
        if val is not None:
            must.append(FieldCondition(key=key, match=MatchValue(value=val)))
    mv("topic", payload.topic)
    mv("country", payload.country)
    mv("sap_release", payload.sap_release)
    mv("vim_release", payload.vim_release)
    mv("content_kind", payload.content_kind)
    mv("language", payload.language)
    mv("customer", payload.customer)
    mv("project", payload.project)
    if payload.tags:
        must.append(FieldCondition(key="tags", match=MatchAny(any=payload.tags)))
    if not must:
        return None
    return Filter(must=must)

# ----------- Endpoints -----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "collection": COLLECTION,
        "embedding_model": EMBEDDING_MODEL,
        "vector_size": VECTOR_SIZE
    }

@app.post("/ingest")
def ingest(item: IngestItem, x_app_key: Optional[str] = Header(default=None, alias="X-App-Key")):
    require_app_key(x_app_key)

    created_at = item.created_at or datetime.utcnow().isoformat()
    text = item.content.strip()
    if not text:
        raise HTTPException(status_code=422, detail="content is required")

    # Dedup by hash of content
    h = sha256(text)
    if item.dedup:
        try:
            # scroll with filter on hash
            flt = Filter(must=[FieldCondition(key="hash", match=MatchValue(value=h))])
            sc, _ = qdrant.scroll(collection_name=COLLECTION, scroll_filter=flt, limit=1, with_payload=False)
            if sc:
                return {"status": "skipped", "reason": "duplicate", "hash": h}
        except Exception:
            # ignore scroll errors; proceed to upsert
            pass

    chunks = chunk_text(text, max_chars=4000)
    points = []
    for ch in chunks:
        vec = embed_text(ch)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "title": item.title,
                    "topic": item.topic,
                    "content": ch,
                    "summary": item.summary,
                    "content_kind": item.content_kind,
                    "language": item.language,
                    "source": item.source,
                    "tags": item.tags,
                    "country": item.country,
                    "sap_release": item.sap_release,
                    "vim_release": item.vim_release,
                    "customer": item.customer,
                    "project": item.project,
                    "created_at": created_at,
                    "hash": h,
                },
            )
        )
    try:
        qdrant.upsert(collection_name=COLLECTION, points=points)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QdrantError: {type(e).__name__}: {str(e)[:300]}")

    return {"status": "ok", "chunks": len(points), "hash": h}

@app.post("/search")
def search(payload: SearchQuery, x_app_key: Optional[str] = Header(default=None, alias="X-App-Key")):
    require_app_key(x_app_key)

    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=422, detail="query is required")

    query_vec = embed_text(payload.query)
    flt = to_filter(payload)
    try:
        hits = qdrant.search(
            collection_name=COLLECTION,
            query_vector=query_vec,
            limit=payload.k,
            query_filter=flt,
            with_payload=True,
            with_vectors=False,
            score_threshold=payload.min_score if payload.min_score else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QdrantError: {type(e).__name__}: {str(e)[:300]}")

    results = []
    for h in hits:
        p = h.payload or {}
        results.append({
            "id": str(h.id),
            "score": h.score,
            "title": p.get("title"),
            "summary": p.get("summary"),
            "content": p.get("content"),
            "topic": p.get("topic"),
            "tags": p.get("tags"),
            "content_kind": p.get("content_kind"),
            "language": p.get("language"),
            "country": p.get("country"),
            "sap_release": p.get("sap_release"),
            "vim_release": p.get("vim_release"),
            "customer": p.get("customer"),
            "project": p.get("project"),
            "source": p.get("source"),
            "created_at": p.get("created_at"),
        })
    return {"results": results}
