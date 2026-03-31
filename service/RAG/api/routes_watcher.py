import logging
from typing import Any, Dict

import httpx
from fastapi import APIRouter, HTTPException

from core.config import get_settings
from models.api_models import WatcherSessionPushRequest, WatcherSessionPushResponse

router = APIRouter(prefix="/watcher", tags=["watcher"])
log = logging.getLogger("rag-service")


def _resolved_watcher_url(override: str | None = None) -> str:
    raw = (override or get_settings().watcher_control_url or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="watcher_control_url is not configured")
    return raw


@router.post("/session", response_model=WatcherSessionPushResponse)
async def push_session_to_watcher(body: WatcherSessionPushRequest):
    session_id = (body.session_id or "").strip()
    if body.enabled and not session_id:
        raise HTTPException(status_code=400, detail="session_id cannot be empty when enabled=true")

    watcher_url = _resolved_watcher_url(body.watcher_url)
    payload: Dict[str, Any] = {
        "session_id": session_id,
        "enabled": bool(body.enabled),
    }

    try:
        async with httpx.AsyncClient(timeout=get_settings().watcher_control_timeout_sec) as client:
            resp = await client.post(watcher_url, json=payload)
        resp.raise_for_status()
    except Exception as e:
        log.warning("watcher session push failed url=%s err=%s", watcher_url, e)
        raise HTTPException(status_code=502, detail=f"watcher push failed: {e}")

    try:
        watcher_response = resp.json()
    except Exception:
        watcher_response = {"raw": resp.text[:500]}

    return WatcherSessionPushResponse(
        ok=True,
        watcher_url=watcher_url,
        session_id=session_id,
        enabled=bool(body.enabled),
        watcher_response=watcher_response,
    )
