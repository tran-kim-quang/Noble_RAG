"""
LightRAG HTTP API Service
------------------------
Endpoints:
  POST /upload-document   - Upload document, save to graph/vector storage
  POST /query             - Query with document context, receive LLM response
  GET  /health            - Health check
  GET  /models            - List available LLM models
"""

import io
import os
import time
import json
import asyncio
import logging
import tempfile
from urllib.parse import urlparse
from typing import Optional, List
from enum import Enum
from functools import partial

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import asyncpg

# LightRAG imports
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import (
    openai_complete_if_cache,
    openai_embed,
    wrap_embedding_func_with_attrs,
)

# ── Config via env vars ────────────────────────────────────────────────
LLM_PROVIDER     = os.getenv("LLM_PROVIDER",       "openai")     # openai, azure, ollama, gemini, claude
LLM_MODEL        = os.getenv("LLM_MODEL",          "gpt-4o-mini")
LLM_API_KEY      = os.getenv("LLM_API_KEY",        "")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", LLM_PROVIDER)
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",    "text-embedding-3-large")
EMBEDDING_DIM    = int(os.getenv("EMBEDDING_DIM",  "3072"))

POSTGRES_URL     = os.getenv("POSTGRES_URL",       "postgresql://rag:rag@localhost:5432/rag_db")
QDRANT_URL       = os.getenv("QDRANT_URL",         "http://localhost:6333")
REDIS_URL        = os.getenv("REDIS_URL",          "redis://localhost:6379/0")

KV_STORAGE       = os.getenv("KV_STORAGE",         "PGKVStorage")
VECTOR_STORAGE   = os.getenv("VECTOR_STORAGE",     "QdrantVectorDBStorage")
GRAPH_STORAGE    = os.getenv("GRAPH_STORAGE",      "PGGraphStorage")
DOC_STATUS_STORAGE = os.getenv("DOC_STATUS_STORAGE", "PGDocStatusStorage")
RAG_WORKSPACE    = os.getenv("RAG_WORKSPACE",      "default")
RAG_WORKING_DIR  = os.getenv("RAG_WORKING_DIR",    "./rag_db")

CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE",     "1024"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP",  "20"))
LOG_LEVEL        = os.getenv("LOG_LEVEL",          "INFO")
QUERY_TIMEOUT_SEC = float(os.getenv("QUERY_TIMEOUT_SEC", "45"))

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("rag-service")


def configure_storage_env() -> None:
    """Map compose-style URLs to LightRAG backend environment variables."""
    parsed_postgres = urlparse(POSTGRES_URL)

    if parsed_postgres.scheme.startswith("postgres"):
        if parsed_postgres.hostname and not os.getenv("POSTGRES_HOST"):
            os.environ["POSTGRES_HOST"] = parsed_postgres.hostname
        if parsed_postgres.port and not os.getenv("POSTGRES_PORT"):
            os.environ["POSTGRES_PORT"] = str(parsed_postgres.port)
        if parsed_postgres.username and not os.getenv("POSTGRES_USER"):
            os.environ["POSTGRES_USER"] = parsed_postgres.username
        if parsed_postgres.password and not os.getenv("POSTGRES_PASSWORD"):
            os.environ["POSTGRES_PASSWORD"] = parsed_postgres.password
        db_name = parsed_postgres.path.lstrip("/")
        if db_name and not os.getenv("POSTGRES_DATABASE"):
            os.environ["POSTGRES_DATABASE"] = db_name

    if QDRANT_URL and not os.getenv("QDRANT_URL"):
        os.environ["QDRANT_URL"] = QDRANT_URL

    if REDIS_URL and not os.getenv("REDIS_URI"):
        os.environ["REDIS_URI"] = REDIS_URL


# ── Enums ──────────────────────────────────────────────────────────────
class StorageType(str, Enum):
    """Storage type for documents"""
    GRAPH = "graph"
    VECTOR = "vector"
    BOTH = "both"


class LLMProviderType(str, Enum):
    """LLM Provider types"""
    OPENAI = "openai"
    CLAUDE = "claude"
    OLLAMA = "ollama"


# ── Pydantic Models ────────────────────────────────────────────────────
class UploadDocumentRequest(BaseModel):
    """Request model for document upload"""
    storage_type: StorageType = StorageType.GRAPH
    metadata: Optional[dict] = None


class QueryRequest(BaseModel):
    """Request model for query"""
    query: str
    top_k: int = 10
    return_structured_output: bool = False


class UploadDocumentResponse(BaseModel):
    """Response model for document upload"""
    status: str
    document_id: str
    chunks_count: int
    storage_type: str
    message: str


class QueryResponse(BaseModel):
    """Response model for query"""
    response: str
    context: List[str]
    tokens_used: Optional[dict] = None
    model: str


class DocumentRecord(BaseModel):
    id: str
    status: Optional[str] = None
    file_path: Optional[str] = None
    content_length: Optional[int] = None
    chunks_count: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Optional[dict] = None


# ── Initialize LLM ─────────────────────────────────────────────────────
def init_llm():
    """Initialize LLM based on provider"""
    log.info(f"Initializing LLM provider='{LLM_PROVIDER}' model='{LLM_MODEL}'...")

    provider_lower = LLM_PROVIDER.lower()

    if provider_lower == "openai":
        async def llm_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            **kwargs,
        ):
            return await openai_complete_if_cache(
                LLM_MODEL,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=LLM_API_KEY,
                base_url=os.getenv("LLM_API_URL") or None,
                **kwargs,
            )

        return llm_func
    elif provider_lower == "ollama":
        from lightrag.llm.ollama import ollama_model_complete

        return partial(
            ollama_model_complete,
            model=LLM_MODEL,
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider for this service build: {LLM_PROVIDER}. "
            "Use 'openai' or 'ollama'."
        )


def init_embedding():
    """Initialize embedding model"""
    log.info(
        f"Initializing embedding provider='{EMBEDDING_PROVIDER}' model='{EMBEDDING_MODEL}'..."
    )

    provider_lower = EMBEDDING_PROVIDER.lower()

    if provider_lower == "ollama":
        from lightrag.llm.ollama import ollama_embed

        @wrap_embedding_func_with_attrs(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=int(os.getenv("MAX_TOKEN_SIZE", "8192")),
            model_name=EMBEDDING_MODEL,
        )
        async def embedding_func(texts: list[str]):
            return await ollama_embed.func(
                texts,
                embed_model=EMBEDDING_MODEL,
                host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                api_key=os.getenv("OLLAMA_API_KEY") or None,
            )

        return embedding_func

    if provider_lower == "openai":
        @wrap_embedding_func_with_attrs(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=int(os.getenv("MAX_TOKEN_SIZE", "8192")),
            model_name=EMBEDDING_MODEL,
        )
        async def embedding_func(texts: list[str]):
            return await openai_embed.func(
                texts,
                model=EMBEDDING_MODEL,
                api_key=LLM_API_KEY,
                base_url=os.getenv("LLM_API_URL") or None,
            )

        return embedding_func
    raise ValueError(
        f"Unsupported embedding provider: {EMBEDDING_PROVIDER}. "
        "Use 'openai' or 'ollama'."
    )


# ── Initialize LightRAG ────────────────────────────────────────────────
log.info("Loading LightRAG components...")
_load_start = time.perf_counter()

try:
    configure_storage_env()
    llm_model_func = init_llm()
    embedding_func = init_embedding()
    
    # Initialize LightRAG instance
    rag = LightRAG(
        working_dir=RAG_WORKING_DIR,
        kv_storage=KV_STORAGE,
        vector_storage=VECTOR_STORAGE,
        graph_storage=GRAPH_STORAGE,
        doc_status_storage=DOC_STATUS_STORAGE,
        workspace=RAG_WORKSPACE,
        llm_model_func=llm_model_func,
        llm_model_name=LLM_MODEL,
        embedding_func=embedding_func,
        chunk_token_size=CHUNK_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP,
    )
    
    _load_time = time.perf_counter() - _load_start
    log.info(f"LightRAG initialized in {_load_time:.2f}s")
    
except Exception as e:
    log.error(f"Failed to initialize LightRAG: {str(e)}")
    raise


# ── FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(
    title="LightRAG API",
    description="Retrieval-Augmented Generation service powered by LightRAG",
    version="1.0.0",
)

QUERY_LOCK = asyncio.Lock()


@app.on_event("startup")
async def startup_event():
    """Initialize LightRAG storages required by latest LightRAG pipeline."""
    await rag.initialize_storages()
    log.info("LightRAG storages initialized")


# ── Health Check ───────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "timestamp": time.time(),
            "service": "rag-service",
            "llm_provider": LLM_PROVIDER,
            "llm_model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
        }
    )


# ── Models Info ────────────────────────────────────────────────────────
@app.get("/models")
async def get_models():
    """Get available LLM models and embedding info"""
    return JSONResponse(
        status_code=200,
        content={
            "llm": {
                "provider": LLM_PROVIDER,
                "model": LLM_MODEL,
                "max_tokens": 2048,
            },
            "embedding": {
                "model": EMBEDDING_MODEL,
                "dimensions": EMBEDDING_DIM,
            },
            "storage": {
                "kv_storage": KV_STORAGE,
                "vector_storage": VECTOR_STORAGE,
                "graph_storage": GRAPH_STORAGE,
                "doc_status_storage": DOC_STATUS_STORAGE,
                "postgres": POSTGRES_URL.replace(
                    POSTGRES_URL.split("@")[0].split(":")[-1], 
                    "****"
                ) if "@" in POSTGRES_URL else POSTGRES_URL,
                "qdrant": QDRANT_URL,
                "redis": REDIS_URL.split("@")[-1] if "@" in REDIS_URL else REDIS_URL,
            }
        }
    )


# ── Document Upload ────────────────────────────────────────────────────
@app.post("/upload-document", response_model=UploadDocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    storage_type: StorageType = Form(StorageType.GRAPH),
    metadata: Optional[str] = Form(None),
):
    """
    Upload a document and process it into graph/vector storage
    
    Args:
        file: Document file (txt, pdf, doc, etc.)
        storage_type: "graph" (default), "vector", or "both"
        metadata: JSON metadata associated with document
    
    Returns:
        UploadDocumentResponse with document_id and chunk count
    """
    
    start_time = time.perf_counter()
    
    try:
        # Read file content
        content = await file.read()
        
        # Decode content
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File must be UTF-8 encoded text"
            )
        
        if not text_content.strip():
            raise HTTPException(
                status_code=400,
                detail="File is empty"
            )
        
        # Parse metadata if provided
        doc_metadata = {}
        if metadata:
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid JSON metadata"
                )
        
        # Add filename and upload timestamp to metadata
        doc_metadata['filename'] = file.filename
        doc_metadata['upload_timestamp'] = time.time()
        
        log.info(
            f"Processing document '{file.filename}' "
            f"storage_type={storage_type.value} size={len(text_content)} chars"
        )
        
        # Process with LightRAG
        # Note: ainsert() always inserts both vector and graph for optimization.
        # Storage type parameter is retained for future extensibility.
        document_id = await rag.ainsert(
            text_content,
        )
        
        # Estimate chunks (rough calculation)
        chunks_count = max(1, len(text_content) // CHUNK_SIZE)
        
        elapsed_time = time.perf_counter() - start_time
        
        log.info(
            f"Document '{file.filename}' processed successfully "
            f"document_id={document_id} chunks={chunks_count} time={elapsed_time:.2f}s"
        )
        
        return UploadDocumentResponse(
            status="success",
            document_id=document_id,
            chunks_count=chunks_count,
            storage_type=storage_type.value,
            message=f"Document processed and stored ({chunks_count} chunks in {elapsed_time:.2f}s)"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error processing document '{file.filename}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


# ── Query Endpoint ─────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with document context
    
    Args:
        request.query: Search query
        request.top_k: Number of top results to use as context
        request.return_structured_output: Return structured data
    
    Returns:
        QueryResponse with LLM answer and context
    """
    
    start_time = time.perf_counter()
    
    try:
        if not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        log.info(f"Processing query: {request.query[:100]}...")

        if QUERY_LOCK.locked():
            raise HTTPException(
                status_code=429,
                detail="RAG is currently processing another query. Please retry shortly.",
            )

        async with QUERY_LOCK:
            try:
                response = await asyncio.wait_for(
                    rag.aquery(
                        request.query,
                        param=QueryParam(
                            top_k=request.top_k,
                            mode="local" if request.return_structured_output else "global",
                        ),
                    ),
                    timeout=QUERY_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                elapsed_time = time.perf_counter() - start_time
                log.error(
                    "Query timed out after %.2fs (timeout=%.2fs). "
                    "Possible LightRAG runtime stall.",
                    elapsed_time,
                    QUERY_TIMEOUT_SEC,
                )
                raise HTTPException(
                    status_code=504,
                    detail=(
                        f"Query timed out after {QUERY_TIMEOUT_SEC:.1f}s. "
                        "RAG runtime may be stalled; retry once or restart rag-service."
                    ),
                )

        # Current LightRAG aquery returns a string answer (or stream iterator when enabled).
        answer = response if isinstance(response, str) else str(response)
        context_items: list[str] = []
        
        elapsed_time = time.perf_counter() - start_time
        
        log.info(
            f"Query processed in {elapsed_time:.2f}s "
            f"model={LLM_MODEL} context_items={len(context_items)}"
        )
        
        return QueryResponse(
            response=answer,
            context=context_items[:request.top_k],
            tokens_used=None,
            model=LLM_MODEL,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


# ── Status Endpoint ────────────────────────────────────────────────────
@app.get("/status")
async def get_status():
    """Get RAG system status"""
    return JSONResponse(
        status_code=200,
        content={
            "service": "rag-service",
            "status": "running",
            "llm_provider": LLM_PROVIDER,
            "llm_model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "storage": {
                "type": "hybrid",
                "graph": GRAPH_STORAGE,
                "vector": VECTOR_STORAGE,
                "session": "Redis",
                "kv": KV_STORAGE,
                "doc_status": DOC_STATUS_STORAGE,
            }
        }
    )


@app.get("/documents")
async def list_documents(limit: int = 50):
    """List uploaded documents from PostgreSQL LightRAG doc status table."""
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")

    parsed = urlparse(POSTGRES_URL)
    if not parsed.scheme.startswith("postgres"):
        raise HTTPException(status_code=500, detail="Invalid POSTGRES_URL")

    db_name = parsed.path.lstrip("/") or "noble_rag"

    try:
        conn = await asyncpg.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            user=parsed.username,
            password=parsed.password,
            database=db_name,
        )
        rows = await conn.fetch(
            """
            SELECT id, status, file_path, content_length, chunks_count, created_at, updated_at, metadata
            FROM lightrag_doc_status
            WHERE workspace = $1
            ORDER BY updated_at DESC NULLS LAST
            LIMIT $2
            """,
            RAG_WORKSPACE,
            limit,
        )
        await conn.close()
    except Exception as e:
        log.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")

    documents = []
    for row in rows:
        metadata = row.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = None

        documents.append(
            {
                "id": row.get("id"),
                "status": row.get("status"),
                "file_path": row.get("file_path"),
                "content_length": row.get("content_length"),
                "chunks_count": row.get("chunks_count"),
                "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
                "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
                "metadata": metadata,
            }
        )

    return JSONResponse(
        {
            "workspace": RAG_WORKSPACE,
            "count": len(documents),
            "documents": documents,
        }
    )


@app.get("/documents/track/{track_id}")
async def get_track_status(track_id: str):
    """Get processing status summary for a LightRAG upload track id."""
    parsed = urlparse(POSTGRES_URL)
    if not parsed.scheme.startswith("postgres"):
        raise HTTPException(status_code=500, detail="Invalid POSTGRES_URL")

    db_name = parsed.path.lstrip("/") or "noble_rag"

    try:
        conn = await asyncpg.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            user=parsed.username,
            password=parsed.password,
            database=db_name,
        )
        rows = await conn.fetch(
            """
            SELECT status, COUNT(*) AS count
            FROM lightrag_doc_status
            WHERE workspace = $1 AND track_id = $2
            GROUP BY status
            """,
            RAG_WORKSPACE,
            track_id,
        )
        await conn.close()
    except Exception as e:
        log.error(f"Failed to get track status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get track status: {e}")

    status_counts: dict[str, int] = {}
    total = 0
    for row in rows:
        status_value = row.get("status") or "unknown"
        count_value = int(row.get("count") or 0)
        status_counts[status_value] = count_value
        total += count_value

    processing = status_counts.get("processing", 0)
    pending = status_counts.get("pending", 0)
    ready = total > 0 and processing == 0 and pending == 0

    return JSONResponse(
        {
            "workspace": RAG_WORKSPACE,
            "track_id": track_id,
            "total": total,
            "ready": ready,
            "status_counts": status_counts,
        }
    )


# ── Run Server ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("RAG_SERVICE_PORT", "8001"))
    host = os.getenv("RAG_SERVICE_HOST", "0.0.0.0")
    
    log.info(f"Starting RAG service on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=LOG_LEVEL.lower(),
    )
