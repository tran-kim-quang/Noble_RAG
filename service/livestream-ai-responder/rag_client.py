"""
RAG Agent client — gửi comment tới Noble RAG Sales Agent và nhận phản hồi.
"""

import asyncio
import logging
from typing import Optional

import httpx

from config import Settings

log = logging.getLogger("livestream-ai-responder.rag")


class RagAgentClient:
    """HTTP client gọi tới rag-service /sales/chat."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.rag_service_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def connect(self) -> None:
        timeout = httpx.Timeout(
            connect=10.0,
            read=self._settings.rag_request_timeout,
            write=10.0,
            pool=10.0,
        )
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)
        log.info("RAG client sẵn sàng → %s", self._base_url)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def ask(
        self,
        message: str,
        session_id: str,
        username: str = "viewer",
    ) -> Optional[str]:
        """
        Gửi `message` tới Sales Agent.

        Trả về chuỗi phản hồi của AI, hoặc None nếu lỗi.
        """
        if not self._client:
            log.error("RAG client chưa được khởi tạo.")
            return None

        payload = {
            "session_id": session_id,
            "message": message,
        }

        for attempt in range(1, self._settings.max_rag_retries + 2):
            try:
                resp = await self._client.post("/sales/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                reply: str = data.get("response", "")
                log.info(
                    "[%s] Q: %s | A: %s",
                    username,
                    message[:80],
                    reply[:120],
                )
                return reply
            except httpx.HTTPStatusError as exc:
                log.warning(
                    "RAG trả lỗi HTTP %s (lần %d): %s",
                    exc.response.status_code,
                    attempt,
                    exc.response.text[:200],
                )
            except Exception as exc:
                log.warning("RAG gọi thất bại (lần %d): %s", attempt, exc)

            if attempt <= self._settings.max_rag_retries:
                await asyncio.sleep(self._settings.retry_delay_sec)

        log.error(
            "Bỏ qua comment sau %d lần thử thất bại.", self._settings.max_rag_retries + 1
        )
        return None
