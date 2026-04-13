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
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings
from core.dependencies import rag
from core.logging import setup_logging, get_logger
from memory.lead_profile_store import ensure_sales_schema
from api.routes_health import router as health_router
from api.routes_documents import router as documents_router
from api.routes_lan_bridge import router as lan_bridge_router
from api.routes_pipeline_v1 import router as pipeline_v1_router
from api.routes_query import router as query_router
from api.routes_sales import router as sales_router
from memory.pipeline_store import ensure_pipeline_schema
from sales.nodes.retrieve_context import warm_retrieval_caches

setup_logging()
log = get_logger("rag-service")
settings = get_settings()

app = FastAPI(
    title="Noble RAG + Sales Agent API",
    description="Haystack-powered retrieval + LangGraph AI Sales Agent for Noble real estate.",
    version="2.0.0",
)

_cors_origins_raw = (settings.cors_origins or "*").strip()
if _cors_origins_raw == "*" or not _cors_origins_raw:
    _cors_origins = ["*"]
else:
    _cors_origins = [item.strip() for item in _cors_origins_raw.split(",") if item.strip()]
_allow_credentials = _cors_origins != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    await rag.initialize_storages()
    await ensure_sales_schema()
    await ensure_pipeline_schema()
    import asyncio
    if settings.warm_retrieval_catalog_on_startup or settings.warm_project_facts_on_startup:
        asyncio.create_task(
            warm_retrieval_caches(
                include_catalog=settings.warm_retrieval_catalog_on_startup,
                include_project_facts=settings.warm_project_facts_on_startup,
                start_delay_sec=settings.warm_retrieval_start_delay_sec,
            )
        )
    log.info("Haystack storages initialised.")

@app.on_event("shutdown")
async def shutdown_event():
    log.info("Đã đóng các kết nối và dọn dẹp ứng dụng.")


app.include_router(health_router)
app.include_router(documents_router)
app.include_router(query_router)
app.include_router(sales_router)
app.include_router(pipeline_v1_router)
app.include_router(lan_bridge_router)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.rag_service_host,
        port=settings.rag_service_port,
        log_level=settings.log_level.lower(),
    )
