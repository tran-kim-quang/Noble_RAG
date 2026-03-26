"""
Livestream AI Responder — main entry point.

Vòng lặp chính:
  1. BRPOP comment từ Redis queue (đẩy bởi livestream-tiktok-service)
  2. Gửi comment tới RAG Sales Agent (/sales/chat)
  3. Log phản hồi (có thể mở rộng để gửi TTS / push notification)
"""

import asyncio
import logging
import signal
import sys

from config import get_settings
from redis_consumer import RedisConsumer
from rag_client import RagAgentClient

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("livestream-ai-responder")


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

async def run_responder() -> None:
    settings = get_settings()
    log.setLevel(settings.log_level.upper())

    redis = RedisConsumer(settings)
    rag = RagAgentClient(settings)

    # Kết nối
    await redis.connect()
    await rag.connect()

    log.info(
        "🚀 Livestream AI Responder sẵn sàng.\n"
        "   Redis queue  : %s → key=%s\n"
        "   RAG Agent    : %s\n"
        "   Session ID   : %s\n"
        "   BRPOP timeout: %ss",
        settings.redis_url,
        settings.redis_queue_key,
        settings.rag_service_url,
        settings.livestream_session_id,
        settings.pop_timeout,
    )

    stop_event = asyncio.Event()

    # Bắt tín hiệu thoát gracefully
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    processed = 0
    skipped = 0

    try:
        while not stop_event.is_set():
            # 1. Lấy comment tiếp theo (blocking BRPOP, tối đa pop_timeout giây)
            comment_obj = await redis.pop_comment()
            if comment_obj is None:
                # Queue rỗng — lặp lại (không tốn CPU vì BRPOP đã block)
                continue

            # 2. Trích xuất thông tin
            comment_text: str = comment_obj.get("comment", "").strip()
            user_info: dict = comment_obj.get("user", {})
            nickname: str = user_info.get("nickname") or user_info.get("unique_id") or "viewer"
            room_id = comment_obj.get("room_id", "unknown")
            enqueued_at = comment_obj.get("enqueued_at", "")

            if not comment_text:
                log.debug("Bỏ qua comment rỗng từ @%s", nickname)
                skipped += 1
                continue

            log.info(
                "📥 Comment #%d | room=%s | @%s: %s",
                processed + skipped + 1,
                room_id,
                nickname,
                comment_text[:120],
            )

            # 3. Gửi tới RAG Sales Agent
            # Session ID bao gồm room để mỗi phòng live có context riêng
            session_id = f"{settings.livestream_session_id}_{room_id}"
            reply = await rag.ask(
                message=comment_text,
                session_id=session_id,
                username=nickname,
            )

            if reply:
                processed += 1
                # In phản hồi rõ ràng — có thể tích hợp TTS / webhook ở đây
                print(
                    f"\n{'─'*60}\n"
                    f"👤 [{nickname}]: {comment_text}\n"
                    f"🤖 [AI]        : {reply}\n"
                    f"{'─'*60}\n",
                    flush=True,
                )
            else:
                skipped += 1

    finally:
        log.info(
            "Dừng Responder. Đã xử lý %d comment, bỏ qua %d.",
            processed,
            skipped,
        )
        await redis.close()
        await rag.close()


if __name__ == "__main__":
    asyncio.run(run_responder())
