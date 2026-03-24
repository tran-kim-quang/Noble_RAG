import asyncio
import time
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
log = logging.getLogger("rag-service")


def now_vietnam_str() -> str:
    return datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S")


def now_vietnam_human() -> str:
    now = datetime.now(VN_TZ)
    return (
        f"{now.hour} giờ {now.minute:02d} phút "
        f"ngày {now.day:02d} tháng {now.month:02d} năm {now.year}"
    )


async def timed_await(label: str, awaitable):
    """Await a coroutine and log elapsed time."""
    started = time.perf_counter()
    try:
        return await awaitable
    finally:
        elapsed = time.perf_counter() - started
        log.info("Timing[%s]: %.2fs", label, elapsed)
