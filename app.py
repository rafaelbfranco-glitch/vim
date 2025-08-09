import os
import uuid
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
)

# OpenAI embeddings
from openai import OpenAI

# ----------- Environment -----------
QDRANT_URL = os.environ.get("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip()
COLLECTION = os.environ.get("COLLECTION_NAME", "vim_knowledge").strip()

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large").strip()
VECTOR_SIZE = int(os.environ.get("VECTOR_SIZE", "3072"))
APP_KEY = os.environ.get("APP_KEY", "").strip()  # optional app-level key for Custom GPT Action

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

ensure_collection()

# ----------- FastAPI -----------
app = FastAPI(title="VIM RAG MVP", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------- Models -----------
class IngestItem(BaseModel):
    topic: Optional[str] = None
    content: str
    source: Optional[str] = None
    tags: Optional[List[str]] = None
    country: Optional[str] = None
    sap_release: Optional[str] = None
    vim_release: Optional[str] = None
    created_at: Optional[str] = None  # ISO8601; will default to now if not provided

class SearchQuery(BaseModel):
    query: str
    k: int = 6
    topic: Optional[str] = None
    country: Optional[str] = None
    sap_release: Optional[str] = None
    vim_release: Optional[str] = None
    tags: Optional[List[str]] = None
    min_score: Optional[float] = 0.0

# ----------- Helpers -----------
def check_app_key(x_app_key: Optional[str] = Header(default=None, convert_underscores=False)):
    """Simple header-based auth for Custom GPT Actions."""
    if APP_KEY:
        if not x_app_key or x_app_key != APP_KEY:
            raise HTTPException(status_code=401, detail="Invalid X-App-Key")
    # if APP_KEY not set, endpoints are open (not recommended for prod).

def embed_text(text: str) -> List[float]:
    resp = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

def chunk_text(text: str, max_chars: int = 2000) -> List[str]:
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

def to_filter(payload: SearchQuery) -> Optional[Filter]:
    must = []
    if payload.topic:
        must.append(FieldCondition(key="topic", match=MatchValue(value=payload.topic)))
    if payload.country:
        must.append(FieldCondition(key="country", match=MatchValue(value=payload.country)))
    if payload.sap_release:
        must.append(FieldCondition(key="sap_release", match=MatchValue(value=payload.sap_release)))
    if payload.vim_release:
        must.append(FieldCondition(key="vim_release", match=MatchValue(value=payload.vim_release)))
    if payload.tags:
        # match any of the provided tags
        must.append(FieldCondition(key="tags", match=MatchAny(any=payload.tags)))
    if not must:
        return None
    return Filter(must=must)

# ----------- Endpoints -----------
@app.get("/health")
def health():
    return {"status": "ok", "collection": COLLECTION}

@app.post("/ingest")
def ingest(item: IngestItem, x_app_key: Optional[str] = Header(default=None, convert_underscores=False)):
    check_app_key(x_app_key)

    created_at = item.created_at or datetime.utcnow().isoformat()
    chunks = chunk_text(item.content, max_chars=2000)

    points = []
    for ch in chunks:
        vec = embed_text(ch)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "topic": item.topic,
                    "content": ch,
                    "source": item.source,
                    "tags": item.tags,
                    "country": item.country,
                    "sap_release": item.sap_release,
                    "vim_release": item.vim_release,
                    "created_at": created_at,
                },
            )
        )
    qdrant.upsert(collection_name=COLLECTION, points=points)
    return {"status": "ok", "chunks": len(points)}

@app.post("/search")
def search(payload: SearchQuery, x_app_key: Optional[str] = Header(default=None, convert_underscores=False)):
    check_app_key(x_app_key)

    query_vec = embed_text(payload.query)
    flt = to_filter(payload)
    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vec,
        limit=payload.k,
        query_filter=flt,
        with_payload=True,
        with_vectors=False,
        score_threshold=payload.min_score if payload.min_score else None,
    )

    results = []
    for h in hits:
        p = h.payload or {}
        results.append({
            "score": h.score,
            "content": p.get("content"),
            "topic": p.get("topic"),
            "tags": p.get("tags"),
            "country": p.get("country"),
            "sap_release": p.get("sap_release"),
            "vim_release": p.get("vim_release"),
            "source": p.get("source"),
            "created_at": p.get("created_at"),
        })
    return {"results": results}
