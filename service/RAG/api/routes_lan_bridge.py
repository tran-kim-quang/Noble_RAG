"""LAN bridge endpoints for Machine A <-> Machine B integration."""

from __future__ import annotations

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.config import get_settings
from integrations.machine_b_face_client import notify_machine_b_delete_face

router = APIRouter(prefix="/api/v1/lan", tags=["lan-bridge"])
vision_router = APIRouter(prefix="/api/v1/vision", tags=["vision-ingest"])
settings = get_settings()
log = logging.getLogger("rag-service")


class VisionRegisterRequest(BaseModel):
    camera_id: str
    stream_url: str
    token: Optional[str] = None
    secret: Optional[str] = None


_vision_cameras: Dict[str, Dict[str, Any]] = {}
_vision_presence_history: Dict[str, list[Dict[str, Any]]] = {}
_vision_pull_tasks: Dict[str, asyncio.Task] = {}
_machine_b_presence_poll_task: Optional[asyncio.Task] = None
_machine_b_heartbeat: Dict[str, Any] = {
    "enabled": True,
    "interval_sec": 2.0,
    "endpoint": "",
    "endpoint_candidates": [],
    "last_poll_at": None,
    "last_ok_at": None,
    "last_error": None,
    "last_event_at": None,
    "total_polls": 0,
    "total_events": 0,
    "consecutive_failures": 0,
}


def _now_ts() -> float:
    return time.time()


def _presence_poll_interval_sec() -> float:
    raw = str(getattr(settings, "machine_b_presence_poll_interval_sec", "2.0") or "2.0").strip()
    try:
        return max(0.5, float(raw))
    except ValueError:
        return 2.0


def _presence_camera_id() -> str:
    raw = str(getattr(settings, "machine_b_presence_camera", "cam01") or "cam01").strip()
    return raw or "cam01"


def _presence_pull_endpoints() -> List[str]:
    candidates: List[str] = []

    raw_primary = str(getattr(settings, "machine_b_presence_latest_path", "") or "").strip()
    raw_status = str(getattr(settings, "machine_b_presence_status_path", "") or "").strip()
    raw_camera = str(getattr(settings, "machine_b_presence_camera_path", "") or "").strip()
    camera_id = _presence_camera_id()

    def _with_camera(path: str) -> str:
        return path.replace("{camera}", camera_id)

    for raw in [
        raw_primary,
        raw_status,
        raw_camera,
        "/api/v1/vision/status",
        "/api/v1/vision/presence/{camera}",
        "/api/v1/vision/presence/latest",
    ]:
        path = _with_camera((raw or "").strip())
        if not path:
            continue
        if not path.startswith("/"):
            path = "/" + path
        if path not in candidates:
            candidates.append(path)

    return candidates


def _ingest_secret() -> str:
    return str(getattr(settings, "machine_a_ingest_secret", "") or "").strip()


def _ingest_allowed_secrets() -> List[str]:
    allowed: List[str] = []
    for candidate in [
        getattr(settings, "machine_a_ingest_secret", ""),
        getattr(settings, "machine_b_api_key", ""),
    ]:
        value = str(candidate or "").strip()
        if value and value not in allowed:
            allowed.append(value)
    return allowed


def _assert_secret(secret: Optional[str]) -> None:
    provided = str(secret or "").strip()
    allowed = _ingest_allowed_secrets()
    if not allowed:
        raise HTTPException(status_code=503, detail="MACHINE_A_INGEST_SECRET is not configured")
    if not provided:
        if bool(getattr(settings, "machine_b_force_update", False)):
            log.warning(
                "vision ingest auth bypass: empty secret allowed because MACHINE_B_FORCE_UPDATE=true"
            )
            return
        log.warning(
            "vision ingest auth failed: provided_len=0 allowed_count=%s",
            len(allowed),
        )
        raise HTTPException(status_code=401, detail="invalid secret")

    if provided not in allowed:
        log.warning(
            "vision ingest auth failed: provided_len=%s allowed_count=%s",
            len(provided),
            len(allowed),
        )
        raise HTTPException(status_code=401, detail="invalid secret")


def _extract_secret_from_headers(request: Request) -> str:
    auth = str(request.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()

    for header_name in [
        "x-ingest-secret",
        "x-ingest-token",
        "x-machine-a-secret",
        "x-machine-a-ingest-secret",
        "x-api-key",
        "x-secret",
    ]:
        value = str(request.headers.get(header_name) or "").strip()
        if value:
            return value

    return ""


def _parse_presence_payload(payload: Any) -> Dict[str, Any]:
    data = payload
    if isinstance(payload, dict) and isinstance(payload.get("event"), dict):
        data = payload.get("event")

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="presence payload must be a JSON object")

    camera = str(data.get("camera") or data.get("camera_id") or "").strip()
    if not camera:
        camera = _presence_camera_id()

    ts_raw = data.get("ts_unix")
    if ts_raw is None:
        ts_raw = data.get("timestamp")
    if ts_raw is None:
        ts_raw = data.get("ts")

    if ts_raw is None:
        ts_unix = _now_ts()
    else:
        try:
            ts_unix = float(ts_raw)
        except Exception:
            raise HTTPException(status_code=400, detail="ts_unix/timestamp must be numeric")
        if ts_unix > 10_000_000_000:
            ts_unix = ts_unix / 1000.0

    vl_obj = data.get("vl") if isinstance(data.get("vl"), dict) else {}

    if "co_nguoi" in data:
        co_nguoi = bool(data.get("co_nguoi"))
    elif "has_person" in data:
        co_nguoi = bool(data.get("has_person"))
    elif "present" in data:
        co_nguoi = bool(data.get("present"))
    elif "detected" in data:
        co_nguoi = bool(data.get("detected"))
    elif "co_nguoi" in vl_obj:
        co_nguoi = bool(vl_obj.get("co_nguoi"))
    elif "has_person" in vl_obj:
        co_nguoi = bool(vl_obj.get("has_person"))
    elif "presence_label_vi" in vl_obj:
        label = str(vl_obj.get("presence_label_vi") or "").strip().lower()
        if "có người" in label or "co nguoi" in label:
            co_nguoi = True
        elif "không" in label or "khong" in label:
            co_nguoi = False
        else:
            raise HTTPException(status_code=400, detail="cannot infer co_nguoi from presence_label_vi")
    else:
        raise HTTPException(status_code=400, detail="co_nguoi/has_person/present is required")

    if "looking_at_camera" in data:
        looking_at_camera: Optional[bool] = bool(data.get("looking_at_camera"))
    elif "look_at_camera" in data:
        looking_at_camera = bool(data.get("look_at_camera"))
    elif "is_looking" in data:
        looking_at_camera = bool(data.get("is_looking"))
    elif "looking_at_camera" in vl_obj:
        looking_at_camera = bool(vl_obj.get("looking_at_camera"))
    elif "look_at_camera" in vl_obj:
        looking_at_camera = bool(vl_obj.get("look_at_camera"))
    elif "is_looking" in vl_obj:
        looking_at_camera = bool(vl_obj.get("is_looking"))
    else:
        # Do not coerce to false when the source omitted gaze field.
        looking_at_camera = None

    secret = str(
        data.get("secret")
        or data.get("ingest_secret")
        or data.get("machine_a_secret")
        or data.get("machine_a_ingest_secret")
        or data.get("api_key")
        or data.get("token")
        or ""
    ).strip()
    return {
        "camera": camera,
        "ts_unix": ts_unix,
        "co_nguoi": co_nguoi,
        "looking_at_camera": looking_at_camera,
        "secret": secret,
    }


async def _pull_mjpeg_loop(camera_id: str) -> None:
    backoff_sec = 1.0
    while True:
        camera = _vision_cameras.get(camera_id)
        if not camera:
            return

        stream_url = str(camera.get("stream_url") or "").strip()
        if not stream_url:
            camera["status"] = "invalid_stream_url"
            await asyncio.sleep(min(backoff_sec, 5.0))
            backoff_sec = min(backoff_sec * 2.0, 30.0)
            continue

        headers: Dict[str, str] = {}
        token = str(camera.get("token") or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            camera["status"] = "connecting"
            async with httpx.AsyncClient(timeout=20.0) as client:
                async with client.stream("GET", stream_url, headers=headers) as resp:
                    if resp.status_code >= 400:
                        camera["status"] = f"error_http_{resp.status_code}"
                        await asyncio.sleep(min(backoff_sec, 5.0))
                        backoff_sec = min(backoff_sec * 2.0, 30.0)
                        continue

                    camera["status"] = "streaming"
                    backoff_sec = 1.0
                    buf = bytearray()

                    async for chunk in resp.aiter_bytes():
                        if not chunk:
                            continue
                        buf.extend(chunk)

                        # Extract JPEG frames from MJPEG byte stream by SOI/EOI markers.
                        while True:
                            start = buf.find(b"\xff\xd8")
                            if start < 0:
                                break
                            end = buf.find(b"\xff\xd9", start + 2)
                            if end < 0:
                                break
                            frame = bytes(buf[start : end + 2])
                            del buf[: end + 2]

                            camera["last_frame"] = frame
                            camera["has_frame"] = True
                            camera["last_seen"] = time.time()
        except asyncio.CancelledError:
            camera = _vision_cameras.get(camera_id)
            if camera:
                camera["status"] = "stopped"
            raise
        except Exception as exc:
            camera = _vision_cameras.get(camera_id)
            if camera:
                camera["status"] = f"error:{type(exc).__name__}"
            await asyncio.sleep(min(backoff_sec, 5.0))
            backoff_sec = min(backoff_sec * 2.0, 30.0)


def _ensure_pull_task(camera_id: str) -> None:
    task = _vision_pull_tasks.get(camera_id)
    if task and not task.done():
        return
    _vision_pull_tasks[camera_id] = asyncio.create_task(_pull_mjpeg_loop(camera_id))


def _presence_history_add(camera_id: str, event: Dict[str, Any], max_items: int = 50) -> None:
    history = _vision_presence_history.setdefault(camera_id, [])
    history.append(event)
    if len(history) > max_items:
        del history[:-max_items]


def _upsert_presence_event(
    *,
    camera_id: str,
    co_nguoi: bool,
    looking_at_camera: Optional[bool],
    ts_unix: float,
    source: str,
    received_at: Optional[float] = None,
) -> None:
    now = _now_ts()
    received = float(received_at if received_at is not None else now)
    camera = _vision_cameras.setdefault(
        camera_id,
        {
            "camera_id": camera_id,
            "stream_url": "",
            "token": "",
            "status": "presence_only",
            "has_frame": False,
            "last_seen": None,
            "last_presence": None,
            "last_looking_at_camera": None,
            "last_presence_ts": None,
            "last_presence_received_at": None,
            "last_presence_source": None,
            "updated_at": now,
            "last_frame": None,
        },
    )
    camera["last_presence"] = bool(co_nguoi)
    if looking_at_camera is None:
        effective_looking = camera.get("last_looking_at_camera")
    else:
        effective_looking = bool(looking_at_camera)
    camera["last_looking_at_camera"] = effective_looking
    camera["last_presence_ts"] = float(ts_unix)
    camera["last_presence_received_at"] = received
    camera["last_presence_source"] = source
    camera["updated_at"] = now
    _presence_history_add(
        camera_id,
        {
            "camera": camera_id,
            "ts_unix": float(ts_unix),
            "received_at": received,
            "co_nguoi": bool(co_nguoi),
            "looking_at_camera": effective_looking,
            "source": source,
        },
    )


def _extract_presence_events(payload: Any) -> List[Dict[str, Any]]:
    def _to_event(item: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        camera = str(item.get("camera") or item.get("camera_id") or "cam01").strip() or "cam01"
        ts_raw = item.get("ts_unix")
        if ts_raw is None:
            ts_raw = _now_ts()
        try:
            ts_val = float(ts_raw)
        except Exception:
            ts_val = _now_ts()
        if ts_val > 10_000_000_000:
            ts_val = ts_val / 1000.0
        if "co_nguoi" in item:
            person = bool(item.get("co_nguoi"))
        elif "has_person" in item:
            person = bool(item.get("has_person"))
        elif "present" in item:
            person = bool(item.get("present"))
        else:
            return None
        if "looking_at_camera" in item:
            looking: Optional[bool] = bool(item.get("looking_at_camera"))
        elif "look_at_camera" in item:
            looking = bool(item.get("look_at_camera"))
        elif "is_looking" in item:
            looking = bool(item.get("is_looking"))
        elif "last_looking_at_camera" in item:
            value = item.get("last_looking_at_camera")
            looking = None if value is None else bool(value)
        else:
            looking = None
        return {"camera": camera, "ts_unix": ts_val, "co_nguoi": person, "looking_at_camera": looking}

    if isinstance(payload, list):
        return [ev for ev in (_to_event(x) for x in payload) if ev is not None]

    if not isinstance(payload, dict):
        return []

    if isinstance(payload.get("events"), list):
        return [ev for ev in (_to_event(x) for x in payload.get("events", [])) if ev is not None]

    if isinstance(payload.get("cameras"), list):
        mapped: List[Dict[str, Any]] = []
        for cam in payload.get("cameras", []):
            if not isinstance(cam, dict):
                continue
            mapped_item = {
                "camera": cam.get("camera_id") or cam.get("camera"),
                "co_nguoi": cam.get("last_presence"),
                "looking_at_camera": cam.get("last_looking_at_camera"),
                "ts_unix": cam.get("last_presence_ts") or cam.get("ts_unix") or _now_ts(),
            }
            event = _to_event(mapped_item)
            if event is not None:
                mapped.append(event)
        return mapped

    if isinstance(payload.get("event"), dict):
        one = _to_event(payload.get("event"))
        return [one] if one else []

    one = _to_event(payload)
    return [one] if one else []


async def _poll_machine_b_presence_loop() -> None:
    interval_sec = _presence_poll_interval_sec()
    _machine_b_heartbeat["interval_sec"] = interval_sec
    endpoints = _presence_pull_endpoints()
    _machine_b_heartbeat["endpoint_candidates"] = endpoints

    while True:
        base = (settings.machine_b_base_url or "").strip().rstrip("/")
        _machine_b_heartbeat["last_poll_at"] = _now_ts()
        _machine_b_heartbeat["total_polls"] = int(_machine_b_heartbeat.get("total_polls") or 0) + 1

        if not base:
            _machine_b_heartbeat["last_error"] = "MACHINE_B_BASE_URL is empty"
            _machine_b_heartbeat["consecutive_failures"] = int(_machine_b_heartbeat.get("consecutive_failures") or 0) + 1
            await asyncio.sleep(interval_sec)
            continue

        try:
            events: List[Dict[str, Any]] = []
            last_http_error: Optional[str] = None
            chosen_endpoint = ""

            async with httpx.AsyncClient(timeout=max(2.0, settings.machine_b_timeout_sec)) as client:
                for endpoint in endpoints:
                    url = f"{base}{endpoint}"
                    resp = await client.get(url, headers=_machine_b_headers())
                    if resp.status_code >= 400:
                        last_http_error = f"HTTP {resp.status_code} from {url}"
                        continue

                    payload: Any
                    try:
                        payload = resp.json()
                    except Exception:
                        payload = {}

                    extracted = _extract_presence_events(payload)
                    chosen_endpoint = endpoint
                    events = extracted
                    break

            if not chosen_endpoint:
                _machine_b_heartbeat["last_error"] = last_http_error or "no successful endpoint response"
                _machine_b_heartbeat["consecutive_failures"] = int(_machine_b_heartbeat.get("consecutive_failures") or 0) + 1
                await asyncio.sleep(interval_sec)
                continue

            _machine_b_heartbeat["endpoint"] = chosen_endpoint
            if events:
                now = _now_ts()
                for event in events:
                    _upsert_presence_event(
                        camera_id=str(event["camera"]),
                        co_nguoi=bool(event["co_nguoi"]),
                        looking_at_camera=event.get("looking_at_camera"),
                        ts_unix=float(event["ts_unix"]),
                        source="machine_b_poll",
                        received_at=now,
                    )
                _machine_b_heartbeat["last_event_at"] = now
                _machine_b_heartbeat["total_events"] = int(_machine_b_heartbeat.get("total_events") or 0) + len(events)

            _machine_b_heartbeat["last_ok_at"] = _now_ts()
            _machine_b_heartbeat["last_error"] = None
            _machine_b_heartbeat["consecutive_failures"] = 0
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _machine_b_heartbeat["last_error"] = f"{type(exc).__name__}: {exc}"
            _machine_b_heartbeat["consecutive_failures"] = int(_machine_b_heartbeat.get("consecutive_failures") or 0) + 1

        await asyncio.sleep(interval_sec)


def start_machine_b_presence_poller() -> None:
    global _machine_b_presence_poll_task
    if _machine_b_presence_poll_task and not _machine_b_presence_poll_task.done():
        return
    _machine_b_presence_poll_task = asyncio.create_task(_poll_machine_b_presence_loop())
    log.info("machine_b presence poller started (every %.2fs)", _presence_poll_interval_sec())


async def stop_machine_b_presence_poller() -> None:
    global _machine_b_presence_poll_task
    task = _machine_b_presence_poll_task
    _machine_b_presence_poll_task = None
    if not task:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    log.info("machine_b presence poller stopped")


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


@vision_router.post("/register")
async def vision_register(request: VisionRegisterRequest):
    _assert_secret(request.secret)
    camera_id = str(request.camera_id or "").strip()
    stream_url = str(request.stream_url or "").strip()
    if not camera_id:
        raise HTTPException(status_code=400, detail="camera_id is required")
    if not stream_url:
        raise HTTPException(status_code=400, detail="stream_url is required")

    _vision_cameras[camera_id] = {
        "camera_id": camera_id,
        "stream_url": stream_url,
        "token": str(request.token or "").strip(),
        "status": "registered",
        "has_frame": False,
        "last_seen": None,
        "last_presence": None,
        "last_looking_at_camera": None,
        "last_presence_ts": None,
        "last_presence_received_at": None,
        "last_presence_source": None,
        "updated_at": time.time(),
        "last_frame": None,
    }
    _ensure_pull_task(camera_id)
    return {"status": "registered", "camera_id": camera_id}


@vision_router.post("/presence")
async def vision_presence(request: Request):
    payload = await request.json()
    parsed = _parse_presence_payload(payload)
    secret = parsed.get("secret") or _extract_secret_from_headers(request)
    _assert_secret(secret)

    camera_id = str(parsed["camera"]).strip()

    camera = _vision_cameras.setdefault(
        camera_id,
        {
            "camera_id": camera_id,
            "stream_url": "",
            "token": "",
            "status": "presence_only",
            "has_frame": False,
            "last_seen": None,
            "last_presence": None,
            "last_looking_at_camera": None,
            "last_presence_ts": None,
            "last_presence_received_at": None,
            "last_presence_source": None,
            "updated_at": time.time(),
            "last_frame": None,
        },
    )
    _upsert_presence_event(
        camera_id=camera_id,
        co_nguoi=bool(parsed["co_nguoi"]),
        looking_at_camera=parsed.get("looking_at_camera"),
        ts_unix=float(parsed["ts_unix"]),
        source="machine_b_push",
        received_at=_now_ts(),
    )
    log.info(
        "vision_presence: camera=%s ts_unix=%s co_nguoi=%s looking_at_camera=%s",
        camera_id,
        float(parsed["ts_unix"]),
        bool(parsed["co_nguoi"]),
        parsed.get("looking_at_camera"),
    )
    return {"status": "ok", "camera": camera_id}


@vision_router.get("/status")
async def vision_status():
    cameras = []
    for camera_id, data in _vision_cameras.items():
        cameras.append(
            {
                "camera_id": camera_id,
                "stream_url": data.get("stream_url"),
                "status": data.get("status"),
                "last_seen": data.get("last_seen"),
                "has_frame": bool(data.get("has_frame")),
                "last_presence": data.get("last_presence"),
                "last_looking_at_camera": data.get("last_looking_at_camera"),
                "last_presence_ts": data.get("last_presence_ts"),
                "last_presence_received_at": data.get("last_presence_received_at"),
                "last_presence_source": data.get("last_presence_source"),
                "seconds_since_last_presence": (
                    None
                    if data.get("last_presence_received_at") is None
                    else max(0.0, _now_ts() - float(data.get("last_presence_received_at")))
                ),
            }
        )
    return {"cameras": cameras, "count": len(cameras)}


@vision_router.get("/heartbeat")
async def vision_heartbeat_status():
    now = _now_ts()
    cameras: Dict[str, Any] = {}
    for camera_id, data in _vision_cameras.items():
        last_recv = data.get("last_presence_received_at")
        cameras[camera_id] = {
            "last_presence": data.get("last_presence"),
            "last_looking_at_camera": data.get("last_looking_at_camera"),
            "last_presence_source": data.get("last_presence_source"),
            "last_presence_received_at": last_recv,
            "seconds_since_last_presence": None if last_recv is None else max(0.0, now - float(last_recv)),
        }

    last_event_at = _machine_b_heartbeat.get("last_event_at")
    return {
        "heartbeat": {
            **_machine_b_heartbeat,
            "seconds_since_last_event": None if last_event_at is None else max(0.0, now - float(last_event_at)),
        },
        "cameras": cameras,
    }


@vision_router.get("/presence/history")
async def vision_presence_history(
    camera: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    camera_id = str(camera or "").strip()
    if camera_id:
        history = list(_vision_presence_history.get(camera_id, []))[-limit:]
        return {"camera": camera_id, "events": history, "count": len(history)}

    merged: list[Dict[str, Any]] = []
    for cid, items in _vision_presence_history.items():
        for item in items[-limit:]:
            merged.append({"camera": cid, **item})
    merged.sort(key=lambda x: float(x.get("ts_unix") or 0.0), reverse=True)
    merged = merged[:limit]
    return {"events": merged, "count": len(merged)}


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
