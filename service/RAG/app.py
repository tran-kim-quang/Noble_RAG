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
  GET  /sales/lead/{session_id}
  PATCH /sales/lead/{session_id}
  GET  /sales/state/{session_id}
  POST /sales/recommendations/refresh
  POST /sales/followup/generate
"""

import uvicorn
from fastapi import FastAPI

from core.config import get_settings
from core.dependencies import rag
from core.logging import setup_logging, get_logger
from api.routes_health import router as health_router
from api.routes_documents import router as documents_router
from api.routes_query import router as query_router
from api.routes_sales import router as sales_router

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
    log.info("LightRAG storages initialised.")


app.include_router(health_router)
app.include_router(documents_router)
app.include_router(query_router)
app.include_router(sales_router)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.rag_service_host,
        port=settings.rag_service_port,
        log_level=settings.log_level.lower(),
    )
