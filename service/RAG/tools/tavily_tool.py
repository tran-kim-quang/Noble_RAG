"""Tavily search tool — thin async wrapper."""

import logging
import httpx
from typing import Optional

from core.config import get_settings

log = logging.getLogger("rag-service")


async def tavily_search(query: str, max_results: int = 5) -> str:
    """Search the web via Tavily API and return a formatted string of results."""
    api_key = get_settings().tavily_api_key
    if not api_key:
        log.warning("TAVILY_API_KEY not set — search unavailable")
        return "Search tool not available (API key missing)."

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return "Không tìm thấy kết quả."
            lines = []
            for r in results:
                title = r.get("title", "")
                content = r.get("content", "")
                url = r.get("url", "")
                lines.append(f"- {title}: {content} ({url})")
            return "\n".join(lines)
    except Exception as e:
        log.error("Tavily search error query='%s': %s", query, e)
        return f"Lỗi khi tìm kiếm: {e}"
