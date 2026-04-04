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
                "pipeline": "vector_only",
                "vector_backend": "qdrant",
                "metadata_backend": "postgres",
                "session_backend": "redis",
                "workspace": settings.rag_workspace,
                "collection": settings.knowledge_collection_name,
                "qdrant": settings.qdrant_url,
                "postgres": settings.postgres_url.split("@")[-1]
                if "@" in settings.postgres_url
                else settings.postgres_url,
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
                "type": "vector_only",
                "vector": "Qdrant",
                "metadata": "PostgreSQL",
                "session": "Redis",
                "workspace": settings.rag_workspace,
                "collection": settings.knowledge_collection_name,
            },
        },
    )
