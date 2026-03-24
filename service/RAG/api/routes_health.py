import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from core.config import get_settings
from core.dependencies import rag

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health")
async def health_check():
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "timestamp": time.time(),
            "service": "rag-service",
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
        },
    )


@router.get("/models")
async def get_models():
    return JSONResponse(
        status_code=200,
        content={
            "llm": {
                "provider": settings.llm_provider,
                "model": settings.llm_model,
                "max_tokens": 2048,
            },
            "embedding": {
                "model": settings.embedding_model,
                "dimensions": settings.embedding_dim,
            },
            "storage": {
                "kv_storage": settings.kv_storage,
                "vector_storage": settings.vector_storage,
                "graph_storage": settings.graph_storage,
                "doc_status_storage": settings.doc_status_storage,
                "qdrant": settings.qdrant_url,
                "redis": settings.redis_url.split("@")[-1]
                if "@" in settings.redis_url
                else settings.redis_url,
            },
        },
    )


@router.get("/status")
async def get_status():
    return JSONResponse(
        status_code=200,
        content={
            "service": "rag-service",
            "status": "running",
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "storage": {
                "type": "hybrid",
                "graph": settings.graph_storage,
                "vector": settings.vector_storage,
                "session": "Redis",
                "kv": settings.kv_storage,
                "doc_status": settings.doc_status_storage,
            },
        },
    )
