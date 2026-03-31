"""
Noble RAG + LangGraph Sales Agent Service
------------------------------------------
Endpoints:
  POST /upload-document
  POST /query/stream
  GET  /documents
  GET  /documents/track/{id}
  DELETE /documents/{id}
  GET  /health  /models  /status
  POST /sales/chat
  POST /sales/chat/stream
  GET  /sales/lead/{session_id}
  PATCH /sales/lead/{session_id}
  GET  /sales/state/{session_id}
  POST /sales/recommendations/refresh
  POST /sales/followup/generate
  POST /sales/session/{session_id}/close
"""

import uvicorn
from fastapi import FastAPI

from core.config import get_settings
from core.dependencies import rag
from core.logging import setup_logging, get_logger
from memory.lead_profile_store import ensure_sales_schema
from api.routes_health import router as health_router
from api.routes_documents import router as documents_router
from api.routes_query import router as query_router
from api.routes_sales import router as sales_router
from api.routes_watcher import router as watcher_router
from sales.nodes.retrieve_context import warm_retrieval_caches

setup_logging()
log = get_logger("rag-service")
settings = get_settings()

app = FastAPI(
    title="Noble RAG + Sales Agent API",
    description="LightRAG-powered retrieval + LangGraph AI Sales Agent for Noble real estate.",
    version="2.0.0",
)


@app.on_event("startup")
async def startup_event():
    await rag.initialize_storages()
    await ensure_sales_schema()
    import asyncio
    asyncio.create_task(warm_retrieval_caches())
    log.info("LightRAG storages initialised.")

@app.on_event("shutdown")
async def shutdown_event():
    log.info("Đã đóng các kết nối và dọn dẹp ứng dụng.")


app.include_router(health_router)
app.include_router(documents_router)
app.include_router(query_router)
app.include_router(sales_router)
app.include_router(watcher_router)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.rag_service_host,
        port=settings.rag_service_port,
        log_level=settings.log_level.lower(),
    )
