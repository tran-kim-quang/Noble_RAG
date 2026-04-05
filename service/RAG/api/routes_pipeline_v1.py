"""API v1 contract endpoints aligned with the Sales RAG + Vision guide."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
import uuid
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.routes_sales import _run_sales_or_search
from core.config import get_settings

log = logging.getLogger("rag-service")
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


def _fold_vn_token(value: str) -> str:
    text = (value or "").strip().lower()
    if not text:
        return ""
    decomp = unicodedata.normalize("NFD", text)
    no_mark = "".join(ch for ch in decomp if unicodedata.category(ch) != "Mn")
    return no_mark.replace("đ", "d")


def _normalize_gender_from_machine_b(raw: str) -> str:
    """Chuẩn hóa giới tính từ Máy B → male | female | unknown (lưu vào gender_guess)."""
    s = (raw or "").strip().lower()
    if not s:
        return "unknown"
    if s in {"1", "01"}:
        return "male"
    if s in {"2", "02"}:
        return "female"
    if s in {"male", "nam", "man", "m", "anh", "ong"}:
        return "male"
    if s in {"female", "nu", "nữ", "woman", "f", "chi", "chị", "co", "cô"}:
        return "female"
    return "unknown"


def _coerce_age_range_from_machine_b(age_token: str) -> str:
    """Map age_band / nhom_tuoi / age_range từ Máy B → nhãn gợi ý (vd. 20-30, 30-45, 45+)."""
    t = (age_token or "").strip()
    if not t:
        return "unknown"
    tl = t.lower().replace("–", "-").replace("—", "-")
    m = re.match(r"^(\d{1,2})\s*-\s*(\d{1,3})$", tl)
    if m:
        return f"{int(m.group(1))}-{m.group(2)}"
    if re.match(r"^\d{1,3}\s*\+\s*$", tl):
        return re.sub(r"\s+", "", tl)
    fold = _fold_vn_token(t)
    if fold in {"young", "tre", "thanh nien", "teen", "adolescent"}:
        return "20-30"
    if fold in {
        "middle_aged",
        "middle",
        "mid",
        "middle-aged",
        "trung nien",
        "trungnien",
        "adult",
    }:
        return "30-45"
    if fold in {"elderly", "old", "senior", "cao tuoi", "caotuoi", "lao nien", "laonien", "gia"}:
        return "45+"
    if t.lower() not in {"unknown", "unk", "none", "null", "n/a"}:
        return t[:80]
    return "unknown"


def _default_vision_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Gom giới + độ tuổi từ JSON Máy B / relay.

    Hợp đồng khuyến nghị (xem docs/LAN_MACHINE_B_INTEGRATION.md):
    - Giới: `gender_guess` (male/female hoặc 1/2) và/hoặc `gioi_tinh` (nam/nữ).
    - Tuổi: `age_range` (vd. 30-45), hoặc `age_band` (young|mid|elderly), hoặc `nhom_tuoi` (tiếng Việt).
    - Máy B (SQLite `customer_faces`): thường đặt `gioi_tinh`, `nhom_tuoi`, `co_nguoi`, `face_score` trong object **`meta`** ở JSON response — luôn đọc `meta` + `metadata`.
    """

    def _pick(*values: Any) -> str:
        for value in values:
            if value is None:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                text = str(int(value))
            else:
                text = str(value).strip()
            # Bỏ qua các giá trị mang nghĩa "không xác định" để tiếp tục tìm ở các key dự phòng khác
            if text and text.lower() not in {"unknown", "unk", "null", "none", "n/a", "không xác định", "chưa rõ", ""}:
                return text
        return ""

    nested = payload.get("vision_summary") if isinstance(payload.get("vision_summary"), dict) else {}
    if not isinstance(nested, dict):
        nested = {}
    vision = payload.get("vision") if isinstance(payload.get("vision"), dict) else {}
    if not isinstance(vision, dict):
        vision = {}
    demographics = payload.get("user_demographics") if isinstance(payload.get("user_demographics"), dict) else {}
    if not isinstance(demographics, dict):
        demographics = {}
    vision_detail = vision.get("detail") if isinstance(vision.get("detail"), dict) else {}
    if not isinstance(vision_detail, dict):
        vision_detail = {}
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

    gender_raw = _pick(
        payload.get("gender_guess"),
        nested.get("gender_guess"),
        payload.get("gioi_tinh"),
        nested.get("gioi_tinh"),
        meta.get("gioi_tinh"),
        meta.get("gender_guess"),
        meta.get("gender_estimate"),
        meta.get("gender"),
        meta.get("sex"),
        metadata.get("gioi_tinh"),
        metadata.get("gender_guess"),
        metadata.get("gender_estimate"),
        metadata.get("gender"),
        metadata.get("sex"),
        payload.get("gender_estimate"),
        nested.get("gender_estimate"),
        payload.get("gender"),
        nested.get("gender"),
        payload.get("sex"),
        nested.get("sex"),
        vision.get("gender_guess"),
        vision.get("gioi_tinh"),
        demographics.get("gender_guess"),
        demographics.get("gioi_tinh"),
        demographics.get("gender"),
        demographics.get("sex"),
        vision_detail.get("gender_guess"),
        vision_detail.get("gioi_tinh"),
        vision_detail.get("gender"),
        vision_detail.get("sex"),
    )
    gender = _normalize_gender_from_machine_b(gender_raw)

    age_band_raw = _pick(
        payload.get("age_band"),
        nested.get("age_band"),
        meta.get("age_band"),
        metadata.get("age_band"),
        vision.get("age_band"),
        demographics.get("age_band"),
        vision_detail.get("age_band"),
    )
    nhom_raw = _pick(
        payload.get("nhom_tuoi"),
        nested.get("nhom_tuoi"),
        meta.get("nhom_tuoi"),
        metadata.get("nhom_tuoi"),
        vision.get("nhom_tuoi"),
        demographics.get("nhom_tuoi"),
        vision_detail.get("nhom_tuoi"),
    )
    age_token = _pick(
        payload.get("age_range"),
        nested.get("age_range"),
        meta.get("age_range"),
        metadata.get("age_range"),
        age_band_raw,
        nhom_raw,
        payload.get("age_group_estimate"),
        nested.get("age_group_estimate"),
        meta.get("age_group_estimate"),
        metadata.get("age_group_estimate"),
        vision.get("age_range"),
        vision.get("age_band"),
        vision.get("nhom_tuoi"),
        vision.get("age_group_estimate"),
        demographics.get("age_range"),
        demographics.get("age_band"),
        demographics.get("nhom_tuoi"),
        demographics.get("age_group_estimate"),
        vision_detail.get("age_range"),
        vision_detail.get("age_band"),
        vision_detail.get("nhom_tuoi"),
    )
    
    log.info(
        "[VISION_PICK] gender_raw=%r age_band_raw=%r nhom_raw=%r age_token=%r",
        gender_raw, age_band_raw, nhom_raw, age_token
    )

    gender = _normalize_gender_from_machine_b(gender_raw)
    age_range = _coerce_age_range_from_machine_b(age_token)

    return {
        "age_range": age_range,
        "age_band": age_band_raw or "unknown",
        "nhom_tuoi": nhom_raw or "unknown",
        "gender_guess": gender,
        "gioi_tinh": "nam" if gender == "male" else ("nu" if gender == "female" else "unknown"),
        "emotion": "unknown",
        "dress_style": "unknown",
        "visible_attributes": [],
        "scene_context": "unknown",
    }


def _vision_context_from_machine_b_register(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Gộp JSON `POST /v1/face/register` (Máy B) → vision_context.

    Hai nhóm bắt buộc cho xưng hô / tư vấn: giới (`gender_guess`/`gioi_tinh`) và tuổi (`age_range`/`age_band`/`nhom_tuoi`).
    Thêm các scalar hữu ích (match_score, …) nếu có — xem docs/LAN_MACHINE_B_INTEGRATION.md.
    """
    merged: Dict[str, Any] = dict(_default_vision_summary(payload))
    passthrough_keys = (
        "match_score",
        "matched",
        "is_existing_customer",
        "is_new_customer",
        "similarity",
        "confidence",
        "quality_score",
        "face_score",
        "co_nguoi",
        "face_detected",
        "status",
        "message",
        "error_code",
    )
    for key in passthrough_keys:
        if key not in payload:
            continue
        val = payload[key]
        if val is None:
            continue
        if isinstance(val, (dict, list)) and key != "detail":
            continue
        merged[key] = val
    for nested_key in ("user_demographics", "demographics", "attributes", "vision", "vision_summary", "meta", "metadata"):
        sub = payload.get(nested_key)
        if isinstance(sub, dict) and sub:
            merged[nested_key] = sub
    return merged


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
    base = _machine_b_base()
    files = {
        "file": (
            filename or "capture.jpg",
            image_bytes,
            content_type or "application/octet-stream",
        )
    }
    force_update = (os.getenv("MACHINE_B_FORCE_UPDATE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    form = {
        "customer_id": customer_id,
        "force_update": "true" if force_update else "false",
    }
    try:
        async with httpx.AsyncClient(timeout=max(20.0, settings.machine_b_timeout_sec)) as client:
            resp = await client.post(
                f"{base}/v1/face/register",
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
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    log.info(
        "machine_b_face_register: POST %s/v1/face/register http_status=%s customer_id=%s response_keys=%s meta_keys=%s",
        base,
        resp.status_code,
        str(payload.get("customer_id") or customer_id),
        sorted(payload.keys()),
        sorted(meta.keys()) if meta else [],
    )
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
    session_id = ""
    reused_session = False

    if incoming_session_id:
        session_id = incoming_session_id
    elif allow_resume:
        latest = await get_latest_active_session_by_customer_id(cid)
        latest_sid = str((latest or {}).get("session_id") or "").strip()
        # Pin history/session key to customer_id for vision-driven conversations.
        if latest_sid and latest_sid == cid:
            session_id = latest_sid
            reused_session = True

    if not session_id:
        session_id = cid

    if not reused_session and not incoming_session_id and allow_resume and session_id == cid:
        existing_row = await load_session_context_row(session_id)
        if existing_row:
            reused_session = True

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
    if isinstance(vision_summary, dict):
        gender_guess = str(
            vision_summary.get("gender_guess")
            or vision_summary.get("gender_estimate")
            or vision_summary.get("gender")
            or vision_summary.get("sex")
            or ""
        ).strip()
        gioi_tinh = str(vision_summary.get("gioi_tinh") or "").strip()
        age_group = str(
            vision_summary.get("age_group_estimate")
            or vision_summary.get("age_band")
            or vision_summary.get("nhom_tuoi")
            or ""
        ).strip()
        if gender_guess:
            lead_profile["gender_guess"] = gender_guess
            lead_profile["gender_estimate"] = gender_guess
        if gioi_tinh:
            lead_profile["gioi_tinh"] = gioi_tinh
        if age_group:
            lead_profile["age_group_estimate"] = age_group
        if vision_summary.get("is_existing_customer") is not None:
            lead_profile["identity_status"] = (
                "matched" if bool(vision_summary.get("is_existing_customer")) else "new"
            )
        if vision_summary.get("matched") is not None and "identity_status" not in lead_profile:
            lead_profile["identity_status"] = "matched" if bool(vision_summary.get("matched")) else "new"
        ms = vision_summary.get("match_score")
        if ms is not None and ms != "":
            try:
                lead_profile["face_match_score"] = float(ms)
            except (TypeError, ValueError):
                lead_profile["face_match_score"] = ms
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
    inner_summary = vision_payload.get("vision_summary")
    vision_summary = _vision_context_from_machine_b_register(vision_payload)
    if isinstance(inner_summary, dict):
        for k, v in inner_summary.items():
            if v not in (None, "", [], "unknown"):
                vision_summary[k] = v

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
    if isinstance(vision_summary, dict):
        gender_guess = str(
            vision_summary.get("gender_guess")
            or vision_summary.get("gender_estimate")
            or vision_summary.get("gender")
            or vision_summary.get("sex")
            or ""
        ).strip()
        gioi_tinh = str(vision_summary.get("gioi_tinh") or "").strip()
        age_group = str(
            vision_summary.get("age_group_estimate")
            or vision_summary.get("age_band")
            or vision_summary.get("nhom_tuoi")
            or ""
        ).strip()
        if gender_guess:
            lead_profile["gender_guess"] = gender_guess
            lead_profile["gender_estimate"] = gender_guess
        if gioi_tinh:
            lead_profile["gioi_tinh"] = gioi_tinh
        if age_group:
            lead_profile["age_group_estimate"] = age_group
        if vision_summary.get("is_existing_customer") is not None:
            lead_profile["identity_status"] = (
                "matched" if bool(vision_summary.get("is_existing_customer")) else "new"
            )
        if vision_summary.get("matched") is not None and "identity_status" not in lead_profile:
            lead_profile["identity_status"] = "matched" if bool(vision_summary.get("matched")) else "new"
        ms = vision_summary.get("match_score")
        if ms is not None and ms != "":
            try:
                lead_profile["face_match_score"] = float(ms)
            except (TypeError, ValueError):
                lead_profile["face_match_score"] = ms
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

    # ── DEBUG: log raw face_payload fields từ Máy B để trace gender
    _fp_gioi_tinh = face_payload.get("gioi_tinh")
    _fp_nhom_tuoi = face_payload.get("nhom_tuoi")
    _fp_face_reg  = face_payload.get("face_registered")
    _fp_co_nguoi  = face_payload.get("co_nguoi")
    _fp_vision    = face_payload.get("vision") or {}
    _fp_vs        = face_payload.get("vision_summary") or {}
    log.info(
        "[DEBUG machine_b raw] face_registered=%s co_nguoi=%s gioi_tinh=%r nhom_tuoi=%r "
        "vision.gioi_tinh=%r vision.nhom_tuoi=%r vision_summary=%r",
        _fp_face_reg, _fp_co_nguoi, _fp_gioi_tinh, _fp_nhom_tuoi,
        _fp_vision.get("gioi_tinh"), _fp_vision.get("nhom_tuoi"),
        dict(list(_fp_vs.items())[:4]) if isinstance(_fp_vs, dict) else _fp_vs,
    )
    print(
        f"[MACHINE_B_RAW] face_registered={_fp_face_reg} co_nguoi={_fp_co_nguoi} "
        f"gioi_tinh={_fp_gioi_tinh!r} nhom_tuoi={_fp_nhom_tuoi!r} "
        f"vision.gioi_tinh={_fp_vision.get('gioi_tinh')!r} "
        f"vision_summary.gender_guess={_fp_vs.get('gender_guess')!r}",
        flush=True,
    )
    machine_b_summary = _vision_context_from_machine_b_register(face_payload)
    if raw_vision_summary:
        vision_summary = dict(raw_vision_summary)
        for key, value in machine_b_summary.items():
            if value in (None, "", "unknown", [], {}):
                continue
            vision_summary[key] = value
    else:
        vision_summary = machine_b_summary
    log.info(
        "chat-with-camera: vision_summary_for_session gender_guess=%s gioi_tinh=%s age_range=%s (sau chuẩn hoá Máy B)",
        vision_summary.get("gender_guess"),
        vision_summary.get("gioi_tinh"),
        vision_summary.get("age_range"),
    )
    print(
        f"[VISION_SUMMARY] gender_guess={vision_summary.get('gender_guess')!r} "
        f"gioi_tinh={vision_summary.get('gioi_tinh')!r} "
        f"age_range={vision_summary.get('age_range')!r} "
        f"nhom_tuoi={vision_summary.get('nhom_tuoi')!r}",
        flush=True,
    )
    # In camera flow, bind conversation history to customer_id from vision result.
    effective_session_id = customer_id
    if requested_session_id and requested_session_id.strip() == customer_id:
        effective_session_id = requested_session_id.strip()

    open_result = await _open_or_resume_session(
        customer_id=customer_id,
        requested_session_id=effective_session_id,
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
