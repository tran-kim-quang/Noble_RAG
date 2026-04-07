"""Tavily search tool — thin async wrapper."""

import logging
from urllib.parse import urlparse

import httpx
from typing import Optional

from core.config import get_settings

log = logging.getLogger("rag-service")


def _normalize_domain(raw: str) -> str:
    value = (raw or "").strip().lower()
    if not value:
        return ""
    if "://" in value:
        parsed = urlparse(value)
        value = parsed.netloc or parsed.path
    return value.strip("/")


def _allowed_domains() -> list[str]:
    cfg = get_settings()
    domains = [_normalize_domain(item) for item in (cfg.tavily_allowed_domains or "").split(",")]
    domains = [item for item in domains if item]
    max_sources = max(1, int(cfg.tavily_max_sources or 2))
    return domains[:max_sources]


async def tavily_search(query: str, max_results: int = 5) -> str:
    """Search the web via Tavily API and return a formatted string of results."""
    api_key = get_settings().tavily_api_key
    if not api_key:
        log.warning("TAVILY_API_KEY not set — search unavailable")
        return "Search tool not available (API key missing)."

    allowed_domains = _allowed_domains()
    effective_max_results = min(max(1, int(max_results)), max(1, int(get_settings().tavily_max_sources or 2)))
    if not allowed_domains:
        log.warning("Tavily search skipped: tavily_allowed_domains is empty")
        return "Search tool not available (tavily_allowed_domains is empty)."

    try:
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": effective_max_results,
            "search_depth": "basic",
        }
        if allowed_domains:
            payload["include_domains"] = allowed_domains

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return "Không tìm thấy kết quả."

            if allowed_domains:
                filtered = []
                for item in results:
                    domain = _normalize_domain(item.get("url") or "")
                    if domain in allowed_domains:
                        filtered.append(item)
                results = filtered

            results = results[:effective_max_results]
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
