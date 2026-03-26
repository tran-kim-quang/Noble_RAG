"""
Redis consumer — đọc comment từ queue của livestream-tiktok-service (BRPOP FIFO).
"""

import asyncio
import json
import logging
from typing import Optional

import redis.asyncio as aioredis

from config import Settings

log = logging.getLogger("livestream-ai-responder.redis")


class RedisConsumer:
    """Async Redis consumer wrapping BRPOP để lấy comment theo thứ tự FIFO."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Optional[aioredis.Redis] = None

    async def connect(self, retries: int = 8, delay: float = 3.0) -> None:
        """Kết nối Redis với auto-retry."""
        for attempt in range(1, retries + 1):
            try:
                self._client = aioredis.from_url(
                    self._settings.redis_url,
                    decode_responses=True,
                )
                await self._client.ping()
                log.info("Kết nối Redis thành công: %s", self._settings.redis_url)
                return
            except Exception as exc:
                log.warning(
                    "Redis kết nối thất bại (lần %d/%d): %s", attempt, retries, exc
                )
                if attempt < retries:
                    await asyncio.sleep(delay)
        raise RuntimeError(
            f"Không thể kết nối Redis sau {retries} lần thử: {self._settings.redis_url}"
        )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
            log.info("Đã đóng kết nối Redis.")

    async def ping(self) -> bool:
        try:
            return bool(self._client and await self._client.ping())
        except Exception:
            return False

    async def pop_comment(self) -> Optional[dict]:
        """
        Lấy comment kế tiếp từ Redis queue (BRPOP, blocking).
        Trả về dict comment, hoặc None nếu timeout / queue rỗng.
        """
        if not self._client:
            return None
        try:
            result = await self._client.brpop(
                self._settings.redis_queue_key,
                timeout=self._settings.pop_timeout,
            )
            if result is None:
                return None  # timeout — queue rỗng
            _key, raw = result
            return json.loads(raw)
        except Exception as exc:
            log.error("Lỗi khi đọc Redis queue: %s", exc)
            return None

    async def queue_length(self) -> int:
        if not self._client:
            return 0
        try:
            return await self._client.llen(self._settings.redis_queue_key)
        except Exception:
            return 0
