"""API v1 contract endpoints aligned with the Sales RAG + Vision guide."""

from __future__ import annotations

import os
import re
import time
import uuid
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.routes_sales import _run_sales_or_search
from core.config import get_settings
from core.dependencies import rag
from memory.lead_profile_store import ensure_sales_schema, load_lead_profile, save_lead_profile
from memory.pipeline_store import (
    create_customer_session,
    ensure_pipeline_schema,
    insert_chat_message,
    insert_vision_context,
    touch_customer_session_activity,
    upsert_document_metadata,
    upsert_session_context_row,
)
from memory.session_context_service import (
    build_and_save_session_context,
    is_session_context_ready,
    merge_runtime_conversation_context,
)
from memory.session_store import get_existing_session_context, load_session_context, save_session_context
from models.api_models import (
    KnowledgeIngestResponse,
    SalesChatV1Request,
    SalesChatV1Response,
    VisionIdentifyAndContextResponse,
)
from sales.nodes.retrieve_context import refresh_retrieval_caches

router = APIRouter(prefix="/api/v1", tags=["pipeline-v1"])
settings = get_settings()


def _slugify_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name or "upload.bin").strip("._")
    return cleaned or "upload.bin"


def _vision_url(path: str) -> str:
    return settings.vision_service_url.rstrip("/") + path


def _assets_dir(kind: str) -> str:
    root = os.path.join(settings.rag_working_dir, "assets", kind)
    os.makedirs(root, exist_ok=True)
    return root


def _default_vision_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    gender = str(payload.get("gender_estimate") or "unknown").strip().lower() or "unknown"
    age_group = str(payload.get("age_group_estimate") or "unknown").strip().lower() or "unknown"
    age_map = {
        "young": "20-30",
        "middle_aged": "30-45",
        "elderly": "45+",
    }
    return {
        "age_range": age_map.get(age_group, "unknown"),
        "gender_guess": gender,
        "emotion": "unknown",
        "dress_style": "unknown",
        "visible_attributes": [],
        "scene_context": "unknown",
    }


def _extract_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except Exception:
        return response.text
    if isinstance(data, dict):
        detail = data.get("detail")
        if isinstance(detail, str):
            return detail
        if isinstance(detail, dict):
            msg = detail.get("message") or detail.get("error_code")
            if isinstance(msg, str):
                return msg
    return str(data)


async def _call_vision_multipart(
    *,
    path: str,
    image: UploadFile,
    image_bytes: bytes,
    form_data: Dict[str, Any],
) -> Dict[str, Any]:
    files = {
        "image": (
            image.filename or "upload.jpg",
            image_bytes,
            image.content_type or "application/octet-stream",
        )
    }
    data = {k: str(v) for k, v in form_data.items() if v is not None}

    async with httpx.AsyncClient(timeout=25.0) as client:
        resp = await client.post(_vision_url(path), data=data, files=files)

    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=_extract_detail(resp))
    try:
        payload = resp.json()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"vision service returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="vision service returned non-object payload")
    return payload


@router.post("/knowledge/ingest", response_model=KnowledgeIngestResponse)
async def knowledge_ingest_v1(
    file: UploadFile = File(...),
    project_id: str = Form(...),
    collection: str = Form(...),
    document_type: str = Form(...),
    source_name: Optional[str] = Form(None),
):
    await ensure_pipeline_schema()

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="file is empty")

    try:
        text_content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="file must be UTF-8 encoded text") from exc
    if not text_content.strip():
        raise HTTPException(status_code=400, detail="file has no readable content")

    document_id = f"doc_{uuid.uuid4().hex[:12]}"
    cleaned_name = _slugify_filename(file.filename or f"{document_id}.txt")
    storage_path = os.path.join(_assets_dir("knowledge"), f"{document_id}_{cleaned_name}")
    with open(storage_path, "wb") as fh:
        fh.write(raw)

    await rag.ainsert(text_content, file_paths=cleaned_name)
    await refresh_retrieval_caches()

    chunks_created = max(1, len(text_content) // max(1, settings.chunk_size))
    await upsert_document_metadata(
        document_id=document_id,
        collection_name=collection,
        project_id=project_id,
        document_type=document_type,
        file_name=source_name or cleaned_name,
        storage_url=storage_path,
        chunk_count=chunks_created,
        ingest_status="indexed",
    )

    return KnowledgeIngestResponse(
        document_id=document_id,
        status="indexed",
        chunks_created=chunks_created,
        collection=collection,
    )


@router.post("/vision/enroll")
async def vision_enroll_v1(
    image: UploadFile = File(...),
    customer_id: str = Form(...),
    source: str = Form("camera"),
    note: Optional[str] = Form(None),
):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="image is empty")
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="only image is accepted")

    payload = await _call_vision_multipart(
        path="/vision/enroll",
        image=image,
        image_bytes=image_bytes,
        form_data={
            "customer_id": customer_id.strip(),
            "source": source.strip(),
            "note": note,
        },
    )
    return JSONResponse(payload)


@router.post("/vision/identify-and-context", response_model=VisionIdentifyAndContextResponse)
async def vision_identify_and_context_v1(
    image: UploadFile = File(...),
    channel: str = Form(...),
    source: str = Form("camera"),
    allow_create_unknown: bool = Form(True),
    client_trace_id: Optional[str] = Form(None),
):
    await ensure_sales_schema()
    await ensure_pipeline_schema()

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="image is empty")
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="only image is accepted")

    session_id = f"ses_{uuid.uuid4().hex[:12]}"
    image_asset_id = f"img_{uuid.uuid4().hex[:12]}"
    image_name = _slugify_filename(image.filename or f"{image_asset_id}.jpg")
    image_storage_path = os.path.join(_assets_dir("vision"), f"{image_asset_id}_{image_name}")
    with open(image_storage_path, "wb") as fh:
        fh.write(image_bytes)

    try:
        vision_payload = await _call_vision_multipart(
            path="/vision/identify-and-context",
            image=image,
            image_bytes=image_bytes,
            form_data={
                "session_id": session_id,
                "channel": channel,
                "source": source,
                "allow_create_unknown": str(bool(allow_create_unknown)).lower(),
                "client_trace_id": client_trace_id,
            },
        )
    except HTTPException as exc:
        detail = str(exc.detail or "")
        detail_lower = detail.lower()
        if "no_face_detected" in detail_lower or "no face" in detail_lower:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "error_code": "no_face_detected",
                    "message": "Khong tim thay khuon mat hop le trong anh",
                },
            )
        if "unknown_face" in detail_lower:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "error_code": "unknown_face",
                    "message": "Khong tim thay khach hang phu hop trong database",
                },
            )
        raise

    customer_id = str(vision_payload.get("customer_id") or "").strip()
    if not customer_id:
        raise HTTPException(status_code=502, detail="vision service did not return customer_id")

    matched = bool(
        vision_payload.get("matched")
        if vision_payload.get("matched") is not None
        else vision_payload.get("is_existing_customer")
    )
    match_score = vision_payload.get("match_score")
    vision_summary = vision_payload.get("vision_summary")
    if not isinstance(vision_summary, dict):
        vision_summary = _default_vision_summary(vision_payload)

    await create_customer_session(
        session_id=session_id,
        customer_id=customer_id,
        channel=channel,
        source=source,
        created_from_vision=True,
    )

    lead_profile = await load_lead_profile(session_id) or {
        "lead_id": session_id,
        "customer_id": customer_id,
        "sales_stage": "new",
        "current_state": "greeting",
        "current_script_step": "S1_opening",
        "context_seeded_by": "vision",
        "context_seeded_at": time.time(),
    }
    lead_profile["customer_id"] = customer_id
    await save_lead_profile(session_id, lead_profile)

    merged_context = await build_and_save_session_context(
        session_id=session_id,
        customer_id=customer_id,
        customer_profile=lead_profile,
        vision_summary=vision_summary,
    )
    await upsert_session_context_row(
        session_id=session_id,
        customer_id=customer_id,
        context_json=merged_context.get("context_json") or {},
    )
    await insert_vision_context(
        session_id=session_id,
        customer_id=customer_id,
        image_asset_id=image_asset_id,
        matched=matched,
        match_score=float(match_score) if isinstance(match_score, (int, float)) else None,
        vision_summary=vision_summary,
    )

    return VisionIdentifyAndContextResponse(
        matched=matched,
        customer_id=customer_id,
        session_id=session_id,
        vision_context_saved=True,
        vision_summary=vision_summary,
    )


@router.post("/sales/chat", response_model=SalesChatV1Response)
async def sales_chat_v1(request: SalesChatV1Request):
    await ensure_sales_schema()
    await ensure_pipeline_schema()

    if request.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported on /api/v1/sales/chat")
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    existing_context = await get_existing_session_context(request.session_id)
    if not is_session_context_ready(existing_context, request.customer_id):
        raise HTTPException(status_code=409, detail="session_context_not_found")
    await create_customer_session(
        session_id=request.session_id,
        customer_id=request.customer_id,
        channel=request.channel,
        source="sales_chat",
        created_from_vision=False,
    )

    result = await _run_sales_or_search(
        session_id=request.session_id,
        user_text=request.message.strip(),
        raw_transcript=None,
    )

    updated_context = await load_session_context(request.session_id)
    if not is_session_context_ready(updated_context, request.customer_id):
        updated_context = await build_and_save_session_context(
            session_id=request.session_id,
            customer_id=request.customer_id,
            customer_profile=await load_lead_profile(request.session_id),
            vision_summary=(updated_context.get("context_json") or {}).get("vision_context"),
            existing_context=updated_context,
        )

    await touch_customer_session_activity(request.session_id)

    response_text = str(result.get("final_response") or "").strip()
    if not response_text:
        response_text = "Xin lỗi, em chưa có đủ dữ liệu để trả lời ngay lúc này."

    context_json = updated_context.get("context_json") or {}
    conversation_context = context_json.get("conversation_context") or {}
    intent = str(result.get("detected_intent") or conversation_context.get("last_intent") or "").strip() or None
    missing_slots = result.get("missing_slots")
    if not isinstance(missing_slots, list):
        missing_slots = conversation_context.get("missing_slots") or []

    sales_stage = str(
        result.get("next_sales_state")
        or (context_json.get("customer_profile") or {}).get("sales_stage")
        or "discovery"
    )

    updated_context = merge_runtime_conversation_context(
        updated_context,
        last_intent=intent,
        missing_slots=list(missing_slots),
        last_question=response_text if missing_slots else None,
    )
    await save_session_context(request.session_id, updated_context)
    await upsert_session_context_row(
        session_id=request.session_id,
        customer_id=request.customer_id,
        context_json=updated_context.get("context_json") or {},
    )

    await insert_chat_message(
        session_id=request.session_id,
        customer_id=request.customer_id,
        role="user",
        content=request.message.strip(),
        intent=intent,
        sales_stage=sales_stage,
    )
    await insert_chat_message(
        session_id=request.session_id,
        customer_id=request.customer_id,
        role="assistant",
        content=response_text,
        intent=intent,
        sales_stage=sales_stage,
    )

    return SalesChatV1Response(
        response=response_text,
        intent=intent,
        sales_stage=sales_stage,
        missing_slots=list(missing_slots),
        context_used=True,
    )
