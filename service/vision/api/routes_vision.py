from fastapi import APIRouter, Header, HTTPException, Request
import httpx

from service.vision.core.config import get_settings
from service.vision.models.vision_models import (
    VisionGenerateRequest,
    VisionGenerateResponse,
)
from service.vision.services.ollama_client import OllamaClient


router = APIRouter()
ollama_client = OllamaClient()


def _normalize_host(host_value: str | None) -> str | None:
    if not host_value:
        return None

    host_only = host_value.split(",")[0].strip().lower()
    host_only = host_only.split(":")[0].strip()
    return host_only or None


@router.post("/generate", response_model=VisionGenerateResponse)
async def generate_vision_response(
    payload: VisionGenerateRequest,
    request: Request,
    x_vision_host: str | None = Header(default=None),
    x_forwarded_host: str | None = Header(default=None),
) -> VisionGenerateResponse:
    settings = get_settings()

    resolved_host = (
        _normalize_host(payload.host_name)
        or _normalize_host(x_vision_host)
        or _normalize_host(x_forwarded_host)
        or _normalize_host(request.url.hostname)
        or "default"
    )
    host_config = settings.resolve(resolved_host)

    try:
        ollama_result = await ollama_client.generate(
            base_url=host_config.ollama_base_url,
            model=host_config.model,
            prompt=payload.prompt,
            images=payload.images,
            stream=payload.stream,
            options=payload.options,
            timeout_seconds=host_config.timeout_seconds,
        )
    except httpx.HTTPStatusError as exc:
        detail = f"Ollama returned {exc.response.status_code}: {exc.response.text}"
        raise HTTPException(status_code=502, detail=detail) from exc
    except httpx.HTTPError as exc:
        detail = f"Could not connect to Ollama at {host_config.ollama_base_url}"
        raise HTTPException(status_code=502, detail=detail) from exc

    return VisionGenerateResponse(
        host_name=resolved_host,
        model=host_config.model,
        ollama_base_url=host_config.ollama_base_url,
        response=str(ollama_result.get("response", "")),
        done=bool(ollama_result.get("done", True)),
        raw=ollama_result,
    )
