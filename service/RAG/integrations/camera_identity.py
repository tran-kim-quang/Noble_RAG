import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncpg
from memory.chat_history_store import load_chat_history, save_chat_history
from core.config import get_settings
from memory.lead_profile_store import load_lead_profile, save_lead_profile
from memory.session_store import load_session_context, save_session_context

log = logging.getLogger("rag-service")


def _resolve_camera_script_path(raw_path: str) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    repo_root = Path(__file__).resolve().parents[3]
    return str((repo_root / raw_path).resolve())


def _resolve_identify_url() -> str:
    settings = get_settings()
    if settings.vision_identify_url:
        return settings.vision_identify_url
    return settings.vision_service_url.rstrip("/") + "/vision/identify"


def _postgres_url() -> str:
    return get_settings().postgres_url


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    if isinstance(value, dict) and len(value) == 0:
        return True
    return False


async def _list_customer_sessions_by_customer_id(
    customer_id: str,
    current_session_id: str,
    limit: int = 5,
) -> List[str]:
    """Read session list from customer mapping table by customer_id.

    This guarantees hydration uses sessions linked to the exact same customer.
    """
    conn = await asyncpg.connect(_postgres_url())
    try:
        rows = await conn.fetch(
            """
            SELECT session_id
            FROM customer.customer_sessions
            WHERE customer_id = $1::uuid AND session_id <> $2
            ORDER BY created_at DESC
            LIMIT $3
            """,
            customer_id,
            current_session_id,
            limit,
        )
    finally:
        await conn.close()
    session_ids: List[str] = []
    for row in rows:
        sid = str(row["session_id"] or "").strip()
        if sid:
            session_ids.append(sid)
    return session_ids


def _merge_lead_profile_missing_fields(current: Dict[str, Any], previous: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(current or {})
    for key, value in (previous or {}).items():
        if key in {"lead_id", "current_state", "current_script_step"}:
            continue
        if _is_empty_value(merged.get(key)) and not _is_empty_value(value):
            merged[key] = value
    return merged


async def maybe_enrich_identity_from_camera(session_id: str) -> Optional[Dict[str, Any]]:
    settings = get_settings()
    if not settings.camera_auto_trigger:
        return None

    session_context = await load_session_context(session_id)
    lead_profile = await load_lead_profile(session_id)

    script_path = _resolve_camera_script_path(settings.camera_action_script_path)
    identify_url = _resolve_identify_url()
    cmd = [
        "python3",
        script_path,
        "--session-id",
        session_id,
        "--camera-index",
        str(settings.camera_index),
        "--identify-url",
        identify_url,
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=settings.camera_trigger_timeout_sec)
    except asyncio.TimeoutError:
        log.warning("camera trigger timeout session=%s", session_id)
        return None
    except Exception as e:
        log.warning("camera trigger failed session=%s err=%s", session_id, e)
        return None

    if proc.returncode != 0:
        err = (stderr_b or b"").decode("utf-8", errors="ignore").strip()
        log.warning("camera script returned non-zero session=%s code=%s err=%s", session_id, proc.returncode, err[:300])
        return None

    raw = (stdout_b or b"").decode("utf-8", errors="ignore").strip()
    if not raw:
        return None

    try:
        payload = json.loads(raw)
    except Exception:
        log.warning("camera script output is not valid JSON session=%s", session_id)
        return None

    session_context["customer_id"] = payload.get("customer_id") or session_context.get("customer_id")
    session_context["identity_status"] = "matched" if payload.get("is_existing_customer") else "new"
    session_context["gender_estimate"] = payload.get("gender_estimate", session_context.get("gender_estimate", "unknown"))
    session_context["age_group_estimate"] = payload.get("age_group_estimate", session_context.get("age_group_estimate", "unknown"))
    lead_profile = lead_profile or {"lead_id": session_id, "current_state": "greeting"}
    lead_profile["customer_id"] = payload.get("customer_id") or lead_profile.get("customer_id")
    lead_profile["identity_status"] = "matched" if payload.get("is_existing_customer") else "new"
    lead_profile["gender_estimate"] = payload.get("gender_estimate", lead_profile.get("gender_estimate", "unknown"))
    lead_profile["age_group_estimate"] = payload.get("age_group_estimate", lead_profile.get("age_group_estimate", "unknown"))

    # Core workflow: customer_id is the durable key that links to previous sessions.
    # If we detected an existing customer, hydrate old lead/chat data into the current session.
    customer_id = str(payload.get("customer_id") or "").strip()
    if customer_id and bool(payload.get("is_existing_customer")):
        previous_session_id: Optional[str] = None
        try:
            candidate_session_ids = await _list_customer_sessions_by_customer_id(
                customer_id=customer_id,
                current_session_id=session_id,
                limit=5,
            )
        except Exception as e:
            log.warning("list customer sessions failed customer_id=%s err=%s", customer_id, e)
            candidate_session_ids = []

        for sid in candidate_session_ids:
            # Guard: if old lead profile has customer_id and mismatches, skip it.
            old_profile = await load_lead_profile(sid) or {}
            old_customer_id = str(old_profile.get("customer_id") or "").strip()
            if old_customer_id and old_customer_id != customer_id:
                continue
            previous_session_id = sid
            break

        if previous_session_id:
            previous_profile = await load_lead_profile(previous_session_id) or {}
            lead_profile = _merge_lead_profile_missing_fields(lead_profile, previous_profile)

            current_history = await load_chat_history(session_id)
            if not current_history:
                previous_history = await load_chat_history(previous_session_id)
                if previous_history:
                    await save_chat_history(session_id, previous_history)
                    session_context["history_hydrated_from_session_id"] = previous_session_id

            session_context["customer_linked_session_id"] = previous_session_id
            session_context["customer_id"] = customer_id

    purchase_history = ((payload.get("customer_context") or {}).get("purchase_history") or [])
    if isinstance(purchase_history, list):
        session_context["customer_purchase_history"] = purchase_history

    await save_session_context(session_id, session_context)
    await save_lead_profile(session_id, lead_profile)

    return payload
