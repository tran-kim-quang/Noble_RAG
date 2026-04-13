from typing import Any

from pydantic import BaseModel, Field


class VisionGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt sent by remote machine.")
    host_name: str | None = Field(
        default=None, description="Optional host name used to resolve runtime config."
    )
    images: list[str] = Field(
        default_factory=list,
        description="Optional base64-encoded image list for multimodal models.",
    )
    stream: bool = Field(
        default=False,
        description="Set true only if caller can parse Ollama stream chunks.",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Ollama generation options (temperature, top_p, etc.).",
    )


class VisionGenerateResponse(BaseModel):
    host_name: str
    model: str
    ollama_base_url: str
    response: str
    done: bool
    raw: dict[str, Any]
