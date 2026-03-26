"""Livestream proxy and status endpoints in RAG Service."""

import logging
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from core.config import get_settings

router = APIRouter(prefix="/livestream", tags=["livestream"])
log = logging.getLogger("rag-service")

# URL của livestream-service trong Docker network
LIVESTREAM_SERVICE_URL = "http://livestream-service:8002"

@router.get("/status")
async def get_livestream_status():
    """
    Proxy request to livestream-service to get current active sessions.
    Allows frontend to call RAG port instead of directly calling livestream port.
    """
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"{LIVESTREAM_SERVICE_URL}/health")
            resp.raise_for_status()
            data = resp.json()
            
            # Extract session IDs for convenience
            active_sessions = []
            for session in data.get("sessions", []):
                uid = session.get("unique_id")
                rid = session.get("room_id")
                active_sessions.append({
                    "unique_id": uid,
                    "room_id": str(rid) if rid else None,
                    "session_id": f"livestream_session_{rid}" if rid else None
                })
            
            return {
                "online": data.get("status") == "ok",
                "auto_watch": data.get("auto_watch"),
                "active_sessions": active_sessions,
                "raw_health": data
            }
        except Exception as e:
            log.error(f"Error connecting to livestream-service: {e}")
            return JSONResponse(
                status_code=503,
                content={"online": False, "error": "Livestream service unavailable"}
            )

@router.get("/latest-answers")
async def get_latest_answers(limit: int = 10):
    """
    Placeholder for a global 'latest 10 livestream Q&A' from all rooms.
    For now, frontend should use /sales/history/{session_id} after getting ID from /status.
    """
    return {"message": "Use /sales/history/{session_id} to get chat history for a specific room"}
