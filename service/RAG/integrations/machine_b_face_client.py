"""HTTP client gọi Machine B (face embedding).

Máy A gửi tới Máy B (khi đủ ``MACHINE_B_BASE_URL`` + ``VISION_FACE_DELETE_TOKEN``)::

    DELETE {MACHINE_B_BASE_URL}/v1/face/{customer_id}
    Header: X-Vision-Face-Delete-Token: <VISION_FACE_DELETE_TOKEN>  # cùng giá trị .env trên A
    Header (tuỳ chọn): X-API-Key: <MACHINE_B_API_KEY>

Các chỗ trên repo gọi xóa face / đồng bộ sau khi xoá dữ liệu phiên-khách:

- ``memory/chat_history_store.purge_all_chat_history`` — purge-all Redis + gọi DELETE từng ``customer_id`` đã gom
- ``api/routes_sales.close_session`` — đóng phiên + DELETE nếu có ``customer_id``
- ``api/routes_sales.notify_machine_b_after_customer_record_removed`` — endpoint bảo vệ token, DELETE một id
- ``api/routes_lan_bridge.machine_b_delete_face`` — proxy DELETE cho admin

**Không** gọi từ ``integrations/camera_identity``: đổi khách chỉ xoá chat/context phiên trên A, không coi là xoá bản ghi user/embedding trên B. Chỉ các luồng **xoá dữ liệu có chủ đích** (đóng phiên, purge-all, gỡ khách, proxy xoá) mới gọi DELETE B.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from urllib.parse import quote

import httpx

from core.config import get_settings

log = logging.getLogger("rag-service")


def _machine_b_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    key = (get_settings().machine_b_api_key or "").strip()
    if key:
        headers["X-API-Key"] = key
    return headers


async def notify_machine_b_delete_face(customer_id: str) -> Optional[Dict[str, Any]]:
    """Gọi Machine B xóa face embedding của customer_id.

    - Trả về dict kết quả nếu HTTP thành công, None nếu không cấu hình / lỗi mạng / 4xx/5xx.
    - Không raise — dùng best-effort (close_session, purge toàn cục).
    """
    cfg = get_settings()
    base = (cfg.machine_b_base_url or "").strip().rstrip("/")
    token = (cfg.vision_face_delete_token or "").strip()
    cid = (customer_id or "").strip()

    if not base or not token or not cid:
        if not base:
            log.debug("notify_machine_b_delete_face: MACHINE_B_BASE_URL not set, skip")
        elif not token:
            log.warning("notify_machine_b_delete_face: VISION_FACE_DELETE_TOKEN not set, skip")
        return None

    url = f"{base}/v1/face/{quote(cid, safe='')}"
    headers = {
        "X-Vision-Face-Delete-Token": token,
        **_machine_b_headers(),
    }
    log.info(
        "machine_b_face: DELETE %s (customer_id=%s, header X-Vision-Face-Delete-Token=from VISION_FACE_DELETE_TOKEN, len=%d)",
        url,
        cid,
        len(token),
    )
    try:
        async with httpx.AsyncClient(timeout=cfg.machine_b_timeout_sec) as client:
            resp = await client.delete(url, headers=headers)
    except httpx.RequestError as e:
        log.warning("notify_machine_b_delete_face: request error customer_id=%s err=%s", cid, e)
        return None

    if resp.status_code >= 400:
        log.warning(
            "notify_machine_b_delete_face: machine B returned %d for customer_id=%s body=%s",
            resp.status_code,
            cid,
            resp.text[:200],
        )
        return None

    try:
        result: Dict[str, Any] = resp.json()
    except Exception:
        result = {"raw": resp.text}

    log.info(
        "machine_b_face: DELETE ok customer_id=%s http_status=%s deleted=%s",
        cid,
        resp.status_code,
        result.get("deleted"),
    )
    return result


async def notify_machine_b_customer_removed(customer_id: str) -> Optional[Dict[str, Any]]:
    """Báo Máy B: `customer_id` đã bị gỡ / quên phía A (DB là khóa chung) — xóa embedding tương ứng."""
    return await notify_machine_b_delete_face(customer_id)
