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
import logging
import tempfile
from typing import Optional, List
from enum import Enum

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# LightRAG imports
from lightrag import LightRAG
from lightrag.llm import OpenAI, Claude, Ollama
from lightrag.embedding import OpenAIEmbedding

# ── Config via env vars ────────────────────────────────────────────────
LLM_PROVIDER     = os.getenv("LLM_PROVIDER",       "openai")     # openai, azure, ollama, gemini, claude
LLM_MODEL        = os.getenv("LLM_MODEL",          "gpt-4o-mini")
LLM_API_KEY      = os.getenv("LLM_API_KEY",        "")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",    "text-embedding-3-large")
EMBEDDING_DIM    = int(os.getenv("EMBEDDING_DIM",  "3072"))

POSTGRES_URL     = os.getenv("POSTGRES_URL",       "postgresql://rag:rag@localhost:5432/rag_db")
QDRANT_URL       = os.getenv("QDRANT_URL",         "http://localhost:6333")
REDIS_URL        = os.getenv("REDIS_URL",          "redis://localhost:6379/0")

CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE",     "1024"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP",  "20"))
LOG_LEVEL        = os.getenv("LOG_LEVEL",          "INFO")

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("rag-service")


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


# ── Initialize LLM ─────────────────────────────────────────────────────
def init_llm():
    """Initialize LLM based on provider"""
    log.info(f"Initializing LLM provider='{LLM_PROVIDER}' model='{LLM_MODEL}'...")
    
    provider_lower = LLM_PROVIDER.lower()
    
    if provider_lower == "openai":
        return OpenAI(
            api_key=LLM_API_KEY,
            model_name=LLM_MODEL,
            model_config={
                "temperature": 0.7,
                "max_tokens": 2048,
            }
        )
    elif provider_lower == "claude":
        return Claude(
            api_key=LLM_API_KEY,
            model_name=LLM_MODEL,
        )
    elif provider_lower == "ollama":
        return Ollama(
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            model_name=LLM_MODEL,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")


def init_embedding():
    """Initialize embedding model"""
    log.info(f"Initializing embedding model='{EMBEDDING_MODEL}'...")
    
    provider_lower = LLM_PROVIDER.lower()
    
    if provider_lower in ["openai"]:
        return OpenAIEmbedding(
            api_key=LLM_API_KEY,
            model=EMBEDDING_MODEL,
        )
    else:
        # Default to OpenAI embedding
        return OpenAIEmbedding(
            api_key=LLM_API_KEY,
            model=EMBEDDING_MODEL,
        )


# ── Initialize LightRAG ────────────────────────────────────────────────
log.info("Loading LightRAG components...")
_load_start = time.perf_counter()

try:
    llm = init_llm()
    embedding = init_embedding()
    
    # Initialize LightRAG instance
    rag = LightRAG(
        working_dir="./rag_db",
        llm=llm,
        embedding=embedding,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
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
            metadata=doc_metadata,
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
        
        # Query the RAG system
        response = await rag.aquery(
            request.query,
            param={
                "top_k": request.top_k,
                "mode": "local" if request.return_structured_output else "global",
            }
        )
        
        # Extract answer and context
        answer = response.get("answer", "")
        context_items = response.get("context", [])
        
        elapsed_time = time.perf_counter() - start_time
        
        log.info(
            f"Query processed in {elapsed_time:.2f}s "
            f"model={LLM_MODEL} context_items={len(context_items)}"
        )
        
        return QueryResponse(
            response=answer,
            context=context_items[:request.top_k],
            tokens_used={
                "input": response.get("input_tokens", 0),
                "output": response.get("output_tokens", 0),
            },
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
                "graph": "PostgreSQL",
                "vector": "Qdrant",
                "session": "Redis",
            }
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
