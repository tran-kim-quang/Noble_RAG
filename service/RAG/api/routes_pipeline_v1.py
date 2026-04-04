"""API v1 contract endpoints aligned with the Sales RAG + Vision guide."""

from __future__ import annotations

import json
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
    get_latest_active_session_by_customer_id,
    insert_chat_message,
    insert_vision_context,
    load_session_context_row,
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
    SalesChatWithCameraV1Response,
    SalesChatV1Request,
    SalesChatV1Response,
    SessionOpenRequest,
    SessionOpenResponse,
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


def _parse_json_object(raw: Optional[str], field_name: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be a valid JSON object") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail=f"{field_name} must be a JSON object")
    return payload


def _machine_b_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    key = (settings.machine_b_api_key or "").strip()
    if key:
        headers["X-API-Key"] = key
    return headers


def _machine_b_base() -> str:
    base = (settings.machine_b_base_url or "").strip().rstrip("/")
    if not base:
        raise HTTPException(status_code=400, detail="MACHINE_B_BASE_URL is not configured")
    return base


async def _register_face_to_machine_b(
    *,
    customer_id: str,
    image_bytes: bytes,
    filename: str,
    content_type: str,
) -> Dict[str, Any]:
    files = {
        "file": (
            filename or "capture.jpg",
            image_bytes,
            content_type or "application/octet-stream",
        )
    }
    form = {"customer_id": customer_id}
    try:
        async with httpx.AsyncClient(timeout=max(20.0, settings.machine_b_timeout_sec)) as client:
            resp = await client.post(
                f"{_machine_b_base()}/v1/face/register",
                data=form,
                files=files,
                headers=_machine_b_headers(),
            )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"cannot reach machine B: {exc!s}") from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=_extract_detail(resp))
    try:
        payload = resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"machine B returned invalid JSON: {exc!s}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="machine B returned non-object payload")
    return payload


async def _open_or_resume_session(
    *,
    customer_id: str,
    requested_session_id: Optional[str],
    allow_resume: bool,
    channel: Optional[str],
    source: Optional[str],
    customer_profile: Optional[Dict[str, Any]],
    vision_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    cid = (customer_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="customer_id is required")

    incoming_session_id = (requested_session_id or "").strip()
    session_id = incoming_session_id
    reused_session = False
    if not session_id and allow_resume:
        latest = await get_latest_active_session_by_customer_id(cid)
        if latest and str(latest.get("session_id") or "").strip():
            session_id = str(latest.get("session_id")).strip()
            reused_session = True
    if not session_id:
        session_id = f"ses_{uuid.uuid4().hex[:12]}"

    await create_customer_session(
        session_id=session_id,
        customer_id=cid,
        channel=channel,
        source=(source or "session_open").strip() or "session_open",
        created_from_vision=False,
    )

    context = await get_existing_session_context(session_id)
    if not context:
        row = await load_session_context_row(session_id)
        if row and isinstance(row.get("context_json"), dict):
            row_payload = {
                "session_id": session_id,
                "customer_id": row.get("customer_id") or cid,
                "context_ready": True,
                "context_json": row.get("context_json") or {},
            }
            await save_session_context(session_id, row_payload)
            context = await get_existing_session_context(session_id)

    lead_profile = await load_lead_profile(session_id) or {
        "lead_id": session_id,
        "customer_id": cid,
        "sales_stage": "new",
        "current_state": "greeting",
        "current_script_step": "S1_opening",
    }
    if customer_profile:
        lead_profile.update(customer_profile)
    lead_profile["customer_id"] = cid
    await save_lead_profile(session_id, lead_profile)

    should_seed_context = (
        not is_session_context_ready(context, cid)
        or bool(customer_profile)
        or isinstance(vision_summary, dict)
    )
    if should_seed_context:
        context = await build_and_save_session_context(
            session_id=session_id,
            customer_id=cid,
            customer_profile=lead_profile,
            vision_summary=vision_summary,
            existing_context=context,
        )
        await upsert_session_context_row(
            session_id=session_id,
            customer_id=cid,
            context_json=context.get("context_json") or {},
        )
    else:
        context = context or await load_session_context(session_id)

    await touch_customer_session_activity(session_id)
    return {
        "session_id": session_id,
        "customer_id": cid,
        "reused_session": bool(reused_session and not incoming_session_id),
        "context": context,
    }


@router.post("/session/open", response_model=SessionOpenResponse)
async def session_open_v1(request: SessionOpenRequest):
    await ensure_sales_schema()
    await ensure_pipeline_schema()
    result = await _open_or_resume_session(
        customer_id=request.customer_id,
        requested_session_id=request.session_id,
        allow_resume=bool(request.allow_resume),
        channel=request.channel,
        source=request.source or "session_open",
        customer_profile=request.customer_profile or {},
        vision_summary=request.vision_summary if isinstance(request.vision_summary, dict) else None,
    )
    context = result["context"]

    return SessionOpenResponse(
        session_id=result["session_id"],
        customer_id=result["customer_id"],
        reused_session=bool(result["reused_session"]),
        context_ready=bool(context.get("context_ready")),
        context_used=True,
    )


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


@router.post("/sales/chat-with-camera", response_model=SalesChatWithCameraV1Response)
async def sales_chat_with_camera_v1(
    file: UploadFile = File(...),
    message: str = Form(...),
    channel: str = Form("kiosk"),
    source: str = Form("machine_b_auto"),
    session_id: Optional[str] = Form(None),
    customer_id_hint: Optional[str] = Form(None),
    allow_resume: bool = Form(True),
    customer_profile_json: Optional[str] = Form(None),
    vision_summary_json: Optional[str] = Form(None),
):
    await ensure_sales_schema()
    await ensure_pipeline_schema()

    user_text = (message or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="image file is empty")

    requested_session_id = (session_id or "").strip() or None
    profile = _parse_json_object(customer_profile_json, "customer_profile_json")
    raw_vision_summary = _parse_json_object(vision_summary_json, "vision_summary_json")

    resolved_hint = (customer_id_hint or "").strip()
    if not resolved_hint and requested_session_id:
        existing_context = await load_session_context(requested_session_id)
        resolved_hint = str(existing_context.get("customer_id") or "").strip()
    if not resolved_hint:
        resolved_hint = f"cam_{uuid.uuid4().hex[:12]}"

    face_payload = await _register_face_to_machine_b(
        customer_id=resolved_hint,
        image_bytes=image_bytes,
        filename=file.filename or f"{resolved_hint}.jpg",
        content_type=file.content_type or "application/octet-stream",
    )

    customer_id = str(face_payload.get("customer_id") or resolved_hint).strip()
    if not customer_id:
        raise HTTPException(status_code=502, detail="machine B did not return customer_id")

    vision_summary = raw_vision_summary or _default_vision_summary(face_payload)
    open_result = await _open_or_resume_session(
        customer_id=customer_id,
        requested_session_id=requested_session_id,
        allow_resume=bool(allow_resume),
        channel=(channel or "").strip() or None,
        source=(source or "").strip() or "machine_b_auto",
        customer_profile=profile,
        vision_summary=vision_summary,
    )
    active_session_id = str(open_result["session_id"]).strip()
    active_customer_id = str(open_result["customer_id"]).strip()

    result = await _run_sales_or_search(
        session_id=active_session_id,
        user_text=user_text,
        raw_transcript=None,
    )

    updated_context = await load_session_context(active_session_id)
    if not is_session_context_ready(updated_context, active_customer_id):
        updated_context = await build_and_save_session_context(
            session_id=active_session_id,
            customer_id=active_customer_id,
            customer_profile=await load_lead_profile(active_session_id),
            vision_summary=(updated_context.get("context_json") or {}).get("vision_context"),
            existing_context=updated_context,
        )

    await touch_customer_session_activity(active_session_id)

    response_text = str(result.get("final_response") or "").strip()
    if not response_text:
        response_text = "Xin loi, em chua co du du lieu de tra loi ngay luc nay."

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
    await save_session_context(active_session_id, updated_context)
    await upsert_session_context_row(
        session_id=active_session_id,
        customer_id=active_customer_id,
        context_json=updated_context.get("context_json") or {},
    )

    await insert_chat_message(
        session_id=active_session_id,
        customer_id=active_customer_id,
        role="user",
        content=user_text,
        intent=intent,
        sales_stage=sales_stage,
    )
    await insert_chat_message(
        session_id=active_session_id,
        customer_id=active_customer_id,
        role="assistant",
        content=response_text,
        intent=intent,
        sales_stage=sales_stage,
    )

    return SalesChatWithCameraV1Response(
        session_id=active_session_id,
        customer_id=active_customer_id,
        reused_session=bool(open_result["reused_session"]),
        response=response_text,
        intent=intent,
        sales_stage=sales_stage,
        missing_slots=list(missing_slots),
        context_used=True,
        face_payload=face_payload,
    )
