"""Backward-compatible query stream alias to sales chat stream."""

import uuid

from fastapi import APIRouter, HTTPException

from api.routes_sales import sales_chat_stream
from models.api_models import QueryRequest, SalesChatRequest

router = APIRouter(tags=["query"])


@router.post("/query/stream")
async def query_rag_stream(request: QueryRequest):
    user_text = (request.query or request.message or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    session_id = (request.session_id or "").strip() or f"sales_stream_{uuid.uuid4().hex}"
    sales_request = SalesChatRequest(
        session_id=session_id,
        message=user_text,
        raw_transcript=None,
    )
    return await sales_chat_stream(sales_request)
