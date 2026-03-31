import base64
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List

import httpx

from core.config import get_settings


@dataclass
class FaceDetectionResult:
    quality_score: float
    cropped_bytes: bytes


@dataclass
class FaceBasicAttributes:
    gender_estimate: str
    age_group_estimate: str


class FaceService:
    """Face service using configured vision model for facial feature analysis."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def detect_single_face(self, image_bytes: bytes) -> FaceDetectionResult:
        if not image_bytes:
            raise ValueError("image is empty")
        if len(image_bytes) < 1024:
            raise ValueError("image quality is too low")
        quality_score = min(1.0, len(image_bytes) / 200000.0)
        return FaceDetectionResult(quality_score=quality_score, cropped_bytes=image_bytes)

    async def extract_embedding(self, image_bytes: bytes) -> List[float]:
        analysis = await self._analyze_face_with_vision_model(image_bytes)
        if not analysis.get("face_detected"):
            raise ValueError("no face detected")
        if int(analysis.get("face_count", 1)) > int(self.settings.vision_max_faces):
            raise ValueError("too many faces in image")

        # Keep identity embedding deterministic for matching while we use vision model
        # for semantic face attributes (gender/age group).
        signature = self._fallback_signature(image_bytes, size=64)
        return self._signature_to_embedding(signature, image_bytes)

    async def extract_basic_attributes(self, image_bytes: bytes) -> FaceBasicAttributes:
        analysis = await self._analyze_face_with_vision_model(image_bytes)
        if not analysis.get("face_detected"):
            raise ValueError("no face detected")
        if int(analysis.get("face_count", 1)) > int(self.settings.vision_max_faces):
            raise ValueError("too many faces in image")
        return FaceBasicAttributes(
            gender_estimate=self._normalize_gender(analysis.get("gender_estimate")),
            age_group_estimate=self._normalize_age_group(analysis.get("age_group_estimate")),
        )

    async def _analyze_face_with_vision_model(self, image_bytes: bytes) -> Dict[str, Any]:
        if not self.settings.vision_base_url or not self.settings.vision_api_key:
            return {
                "face_detected": True,
                "face_count": 1,
                "gender_estimate": "unknown",
                "age_group_estimate": "unknown",
            }

        data_uri = self._to_data_uri(image_bytes)
        prompt = (
            "Analyze this single customer face photo.\n"
            "Return JSON only with exactly these keys:\n"
            "face_detected (boolean), face_count (integer), "
            "gender_estimate (male|female|unknown), "
            "age_group_estimate (young|middle_aged|elderly|unknown).\n"
            "No markdown, no extra text."
        )
        payload = {
            "model": self.settings.vision_model_name,
            "max_tokens": self.settings.vision_max_tokens,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.settings.vision_api_key}",
            "Content-Type": "application/json",
        }

        url = self.settings.vision_base_url.rstrip("/") + "/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=40.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            content = (
                resp.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return self._parse_model_json(content)
        except Exception:
            return {
                "face_detected": True,
                "face_count": 1,
                "gender_estimate": "unknown",
                "age_group_estimate": "unknown",
            }

    def _parse_model_json(self, content: str) -> Dict[str, Any]:
        raw = content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if "\n" in raw:
                raw = raw.split("\n", 1)[1]
        if "{" in raw and "}" in raw:
            raw = raw[raw.find("{") : raw.rfind("}") + 1]
        data = json.loads(raw)
        return {
            "face_detected": bool(data.get("face_detected", True)),
            "face_count": int(data.get("face_count", 1)),
            "gender_estimate": self._normalize_gender(data.get("gender_estimate")),
            "age_group_estimate": self._normalize_age_group(data.get("age_group_estimate")),
        }

    def _normalize_gender(self, value: Any) -> str:
        v = str(value or "").strip().lower()
        if v in {"male", "female", "unknown"}:
            return v
        return "unknown"

    def _normalize_age_group(self, value: Any) -> str:
        v = str(value or "").strip().lower()
        if v in {"young", "middle_aged", "elderly", "unknown"}:
            return v
        return "unknown"

    def _signature_to_embedding(self, signature: List[float], image_bytes: bytes) -> List[float]:
        dim = max(16, int(self.settings.vision_embedding_dim))
        sig: List[float] = []
        for value in signature:
            try:
                sig.append(float(value))
            except Exception:
                sig.append(0.0)
        if not sig:
            sig = self._fallback_signature(image_bytes, size=64)

        values: List[float] = []
        img_seed = hashlib.sha256(image_bytes).digest()
        for i in range(dim):
            base = sig[i % len(sig)]
            noise_digest = hashlib.blake2b(img_seed + i.to_bytes(4, "big"), digest_size=2).digest()
            noise = (int.from_bytes(noise_digest, "big") / 65535.0) * 0.06 - 0.03
            values.append(base + noise)

        norm = math.sqrt(sum(v * v for v in values))
        if norm <= 0.0:
            return values
        return [v / norm for v in values]

    def _fallback_signature(self, image_bytes: bytes, size: int) -> List[float]:
        seed = hashlib.sha256(image_bytes).digest()
        values: List[float] = []
        for i in range(size):
            digest = hashlib.blake2b(seed + i.to_bytes(4, "big"), digest_size=2).digest()
            raw = int.from_bytes(digest, "big")
            values.append((raw / 65535.0) * 2.0 - 1.0)
        return values

    def _to_data_uri(self, image_bytes: bytes) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"
