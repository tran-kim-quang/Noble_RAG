"""LAN bridge endpoints for Machine A <-> Machine B integration."""

from __future__ import annotations

import logging
from typing import Any, Dict
from urllib.parse import quote

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from core.config import get_settings
from integrations.machine_b_face_client import notify_machine_b_delete_face

router = APIRouter(prefix="/api/v1/lan", tags=["lan-bridge"])
settings = get_settings()
log = logging.getLogger("rag-service")


def _machine_b_base() -> str:
    base = (settings.machine_b_base_url or "").strip().rstrip("/")
    if not base:
        raise HTTPException(
            status_code=400,
            detail="MACHINE_B_BASE_URL is not configured on Machine A",
        )
    return base


def _machine_b_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    key = (settings.machine_b_api_key or "").strip()
    if key:
        headers["X-API-Key"] = key
    return headers


def _request_error_detail(e: httpx.RequestError) -> str:
    url = str(getattr(getattr(e, "request", None), "url", "") or "")
    return f"cannot reach machine B ({type(e).__name__}) url={url}: {e!s}"


@router.get("/machine-b/health")
async def machine_b_health():
    base = _machine_b_base()
    try:
        async with httpx.AsyncClient(timeout=settings.machine_b_timeout_sec) as client:
            resp = await client.get(f"{base}/health", headers=_machine_b_headers())
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=_request_error_detail(e)) from e
    payload: Any
    try:
        payload = resp.json()
    except Exception:
        payload = resp.text
    return JSONResponse(
        status_code=200,
        content={
            "machine_b_url": base,
            "reachable": resp.status_code < 400,
            "status_code": resp.status_code,
            "payload": payload,
        },
    )


@router.get("/machine-b/face/{customer_id}")
async def machine_b_get_face(customer_id: str):
    base = _machine_b_base()
    cid = (customer_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="customer_id is required")
    path_customer_id = quote(cid, safe="")

    async with httpx.AsyncClient(timeout=settings.machine_b_timeout_sec) as client:
        try:
            resp = await client.get(
                f"{base}/v1/face/{path_customer_id}",
                headers=_machine_b_headers(),
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=_request_error_detail(e)) from e
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    try:
        payload = resp.json()
    except Exception:
        payload = {"raw": resp.text}
    return JSONResponse(status_code=200, content=payload)


@router.delete("/machine-b/face/{customer_id}")
async def machine_b_delete_face(customer_id: str):
    """Proxy xóa face embedding trên Machine B cho customer_id.

    Machine B cần VISION_FACE_DELETE_TOKEN được cấu hình trên cả hai đầu.
    """
    cid = (customer_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="customer_id is required")
    result = await notify_machine_b_delete_face(cid)
    if result is None:
        cfg = get_settings()
        if not cfg.machine_b_base_url:
            raise HTTPException(status_code=400, detail="MACHINE_B_BASE_URL is not configured")
        if not cfg.vision_face_delete_token:
            raise HTTPException(status_code=503, detail="VISION_FACE_DELETE_TOKEN is not configured")
        raise HTTPException(status_code=502, detail="Machine B did not respond successfully")
    return JSONResponse(status_code=200, content=result)


@router.post("/machine-b/face/register")
async def machine_b_register_face(
    customer_id: str = Form(...),
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    center_w: float = Form(0.70),
    center_h: float = Form(0.80),
    margin: int = Form(8),
):
    base = _machine_b_base()
    cid = (customer_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="customer_id is required")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="image file is empty")

    params = {
        "threshold": threshold,
        "center_w": center_w,
        "center_h": center_h,
        "margin": margin,
    }
    files = {
        "file": (
            file.filename or "upload.jpg",
            image_bytes,
            file.content_type or "application/octet-stream",
        )
    }
    data = {"customer_id": cid}

    try:
        async with httpx.AsyncClient(timeout=max(20.0, settings.machine_b_timeout_sec)) as client:
            resp = await client.post(
                f"{base}/v1/face/register",
                params=params,
                data=data,
                files=files,
                headers=_machine_b_headers(),
            )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=_request_error_detail(e)) from e
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    try:
        payload = resp.json()
    except Exception:
        payload = {"raw": resp.text}
    return JSONResponse(status_code=200, content=payload)
