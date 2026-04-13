"""Tavily search tool - thin async wrapper."""

import logging
import time
from typing import Optional

import httpx

from core.config import get_settings

log = logging.getLogger("rag-service")


async def tavily_search(query: str, max_results: int = 5) -> str:
    """Search the web via Tavily API and return a formatted string of results."""
    api_key = get_settings().tavily_api_key
    if not api_key:
        log.warning("TAVILY_API_KEY not set - search unavailable")
        return "Search tool not available (API key missing)."

    t0 = time.perf_counter()
    log.info("Tavily request start: query='%s' max_results=%s", query, max_results)
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
            log.info(
                "Tavily request done: query='%s' results=%s latency=%.3fs",
                query,
                len(results),
                time.perf_counter() - t0,
            )
            if not results:
                return "Khong tim thay ket qua."
            lines = []
            for r in results:
                title = r.get("title", "")
                content = r.get("content", "")
                url = r.get("url", "")
                lines.append(f"- {title}: {content} ({url})")
            return "\n".join(lines)
    except Exception as e:
        log.error("Tavily search error query='%s': %s", query, e)
        return f"Loi khi tim kiem: {e}"
