from typing import Any

import httpx


class OllamaClient:
    async def generate(
        self,
        *,
        base_url: str,
        model: str,
        prompt: str,
        images: list[str],
        stream: bool,
        options: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        if images:
            payload["images"] = images
        if options:
            payload["options"] = options

        endpoint = f"{base_url.rstrip('/')}/api/generate"
        timeout = httpx.Timeout(timeout_seconds)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
