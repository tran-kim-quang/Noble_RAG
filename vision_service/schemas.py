from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator


class VisionIdentifyRequest(BaseModel):
    image_base64: str = Field(min_length=1)
    image_filename: str | None = None
    image_content_type: str | None = None

    @field_validator("image_base64")
    @classmethod
    def validate_image_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("image_base64 must not be blank")
        return cleaned


class VisionProfile(BaseModel):
    recognized: bool = False
    name: str | None = None
    age: int | None = Field(default=None, ge=0, le=120)
    gender: Literal["male", "female"] | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: Literal["face_db", "local_face_analysis", "vlm", "none", "error"] = "none"
    face_count: int = Field(default=0, ge=0)
    bbox: list[int] | None = None
    reason: str | None = None
    face_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    people_count: int
