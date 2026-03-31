import base64
import json
from dataclasses import dataclass
from typing import Any, Dict, List

import httpx
import cv2
from common.opencv_face_runtime import OpenCVFaceRuntime, quality_from_face_box

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
    """Face service using YuNet + SFace for detection/embedding and VLM for attributes."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.runtime = OpenCVFaceRuntime(cache_dir=self.settings.vision_model_cache_dir)

    async def detect_single_face(self, image_bytes: bytes) -> FaceDetectionResult:
        if not image_bytes:
            raise ValueError("image is empty")
        if len(image_bytes) < 1024:
            raise ValueError("image quality is too low")
        frame = self._decode_image(image_bytes)
        face = self._select_single_face(frame)
        if face is None:
            raise ValueError("no face detected")
        cropped = self.runtime.crop_face_jpeg(frame, face)
        quality_score = quality_from_face_box(face, frame.shape)
        return FaceDetectionResult(quality_score=quality_score, cropped_bytes=cropped)

    async def extract_embedding(self, image_bytes: bytes) -> List[float]:
        frame = self._decode_image(image_bytes)
        face = self._select_single_face(frame)
        if face is None:
            raise ValueError("no face detected")
        feature = self.runtime.extract_feature(frame, face)
        return self.runtime.to_list(feature)

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

    def _to_data_uri(self, image_bytes: bytes) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _decode_image(self, image_bytes: bytes):
        import numpy as np

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise ValueError("image cannot be decoded")
        return frame

    def _select_single_face(self, frame) -> Any:
        min_face_size = float(self.settings.vision_min_face_size)
        detected_faces = self.runtime.detect_faces(frame)
        if not detected_faces:
            raise ValueError("no face detected")
        faces = [
            face for face in detected_faces if min(face.width, face.height) >= min_face_size
        ]
        if not faces:
            raise ValueError("face is too small")
        if len(faces) > int(self.settings.vision_max_faces):
            raise ValueError("too many faces in image")
        return faces[0]
