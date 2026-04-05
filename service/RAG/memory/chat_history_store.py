"""Chat history storage backed by Redis.

Key pattern: chat_history:{session_id}
TTL: 24 hours (86400 s)
Max messages kept: HISTORY_MAX_TURNS * 2 (user + assistant)
"""

import logging
from typing import Any, Dict, List, Set, Tuple

from core.config import get_settings
from memory.local_snapshot_store import delete_all_live_session_snapshots, load_local_session_snapshot
from memory.pipeline_store import purge_all_sales_session_tables_postgres, purge_sales_db_for_session_ids
from memory.redis_store import (
    redis_delete,
    redis_delete_keys_matching_pattern,
    redis_get_json,
    redis_scan_keys_matching,
    redis_set_json,
)

HISTORY_MAX_TURNS = 10
log = logging.getLogger("rag-service")


def _redis_url() -> str:
    return get_settings().redis_url


def _key(session_id: str) -> str:
    return f"chat_history:{session_id}"


async def load_chat_history(session_id: str) -> List[Dict[str, Any]]:
    data = await redis_get_json(_redis_url(), _key(session_id))
    if isinstance(data, list):
        log.debug("chat_history loaded: session=%s msgs=%d", session_id, len(data))
        return data
    snapshot = load_local_session_snapshot(session_id)
    history = snapshot.get("chat_history")
    if isinstance(history, list):
        return history
    return []


async def save_chat_history(
    session_id: str,
    history: List[Dict[str, Any]],
    max_turns: int = HISTORY_MAX_TURNS,
) -> None:
    trimmed = history[-(max_turns * 2):]
    await redis_set_json(_redis_url(), _key(session_id), trimmed)
    log.debug("chat_history saved: session=%s msgs=%d", session_id, len(trimmed))


async def append_turn(
    session_id: str,
    user_text: str,
    assistant_text: str,
    max_turns: int = HISTORY_MAX_TURNS,
) -> List[Dict[str, Any]]:
    history = await load_chat_history(session_id)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": assistant_text})
    await save_chat_history(session_id, history, max_turns)
    return history


async def delete_chat_history(session_id: str) -> bool:
    return await redis_delete(_redis_url(), _key(session_id))


async def _scan_redis_session_keys_and_customer_ids(url: str) -> Tuple[Set[str], List[str]]:
    """Trước khi xóa Redis: mọi session_id (hậu tố key) + customer_id trong JSON (cho Máy B)."""
    all_ids: Set[str] = set()
    customer_ids: Set[str] = set()
    for prefix in ("chat_history:", "session_context:", "lead_profile_cache:"):
        pl = len(prefix)
        async for key in redis_scan_keys_matching(url, prefix + "*"):
            if not key.startswith(prefix):
                continue
            suffix = key[pl:].strip()
            if suffix:
                all_ids.add(suffix)
            if prefix in ("session_context:", "lead_profile_cache:"):
                data = await redis_get_json(url, key)
                if isinstance(data, dict):
                    cid = str(data.get("customer_id") or "").strip()
                    if cid:
                        all_ids.add(cid)
                        customer_ids.add(cid)
    return all_ids, sorted(customer_ids)


async def purge_all_chat_history(*, notify_machine_b: bool = True) -> Dict[str, Any]:
    """Xóa sạch cache phiên sales: Redis, Postgres (lead + customer_sessions CASCADE), snapshot local.

    Trước đây chỉ xóa Redis nên ``load_lead_profile`` / ``load_session_context_row`` nạp lại dữ liệu cũ
    từ Postgres + file ``live_snapshots`` — RAG vẫn như \"khách cũ\" và vision có thể không khớp Máy B.

    Trước khi xóa Redis: quét key để lấy session_id + customer_id. Xóa hàng Postgres tương ứng, rồi Redis,
    rồi xóa toàn bộ ``*.txt`` trong thư mục live_snapshots. Nếu Redis đã trống, truncate bảng phiên Postgres
    và gom ``customer_id`` từ đó để vẫn có thể gọi Máy B.
    """
    url = _redis_url()
    all_session_ids, customer_ids_from_redis = await _scan_redis_session_keys_and_customer_ids(url)

    if all_session_ids:
        pg_stats = await purge_sales_db_for_session_ids(sorted(all_session_ids))
        customer_ids = list(customer_ids_from_redis)
    else:
        pg_stats, customer_ids = await purge_all_sales_session_tables_postgres()

    redis_chat = await redis_delete_keys_matching_pattern(url, "chat_history:*")
    redis_session = await redis_delete_keys_matching_pattern(url, "session_context:*")
    redis_lead = await redis_delete_keys_matching_pattern(url, "lead_profile_cache:*")
    snap_deleted = delete_all_live_session_snapshots()
    log.info(
        "purge_all_chat_history: pg=%s chat=%d session_ctx=%d lead=%d snapshots_deleted=%d customers_b=%s",
        pg_stats,
        redis_chat,
        redis_session,
        redis_lead,
        snap_deleted,
        customer_ids,
    )
    total_redis = redis_chat + redis_session + redis_lead

    machine_b_block: Dict[str, Any] = {
        "notify_attempted": False,
        "results": [],
        "skipped_reason": None,
    }

    if not notify_machine_b:
        machine_b_block["skipped_reason"] = "notify_machine_b=false"
    elif not customer_ids:
        machine_b_block["skipped_reason"] = "no_customer_ids_for_machine_b_notify"
    else:
        from core.config import get_settings
        from integrations.machine_b_face_client import notify_machine_b_delete_face

        cfg = get_settings()
        if not cfg.purge_all_notify_machine_b:
            machine_b_block["skipped_reason"] = "disabled_by_PURGE_ALL_NOTIFY_MACHINE_B=false"
        elif not (cfg.machine_b_base_url or "").strip():
            machine_b_block["skipped_reason"] = "MACHINE_B_BASE_URL not set"
        elif not (cfg.vision_face_delete_token or "").strip():
            machine_b_block["skipped_reason"] = "VISION_FACE_DELETE_TOKEN not set"
        else:
            machine_b_block["notify_attempted"] = True
            for cid in customer_ids:
                detail = await notify_machine_b_delete_face(cid)
                machine_b_block["results"].append(
                    {
                        "customer_id": cid,
                        "ok": detail is not None,
                        "machine_b": detail,
                    }
                )

    return {
        **pg_stats,
        "redis_chat_history_keys_deleted": redis_chat,
        "redis_session_context_keys_deleted": redis_session,
        "redis_lead_profile_cache_keys_deleted": redis_lead,
        "redis_keys_deleted": total_redis,
        "live_snapshot_files_deleted": snap_deleted,
        "session_ids_seen_in_redis_before_purge": sorted(all_session_ids) if all_session_ids else [],
        "customer_ids_collected_before_purge": customer_ids,
        "machine_b": machine_b_block,
    }
