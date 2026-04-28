from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from vision_service.config import Settings
from vision_service.schemas import VisionProfile

log = logging.getLogger("vision-service")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class VisionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.face_db_dir = Path(settings.face_db_dir).resolve()

        import cv2  # lazy import for environments that do not run vision service
        import numpy as np
        from insightface.app import FaceAnalysis

        self.cv2 = cv2
        self.np = np

        providers = ["CPUExecutionProvider"]
        ctx_id = -1
        if settings.use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0

        self.app = FaceAnalysis(name=settings.model_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(settings.det_size, settings.det_size))
        self.people: dict[str, Any] = {}
        self.image_count: dict[str, int] = {}
        self.load()

    def load(self) -> None:
        if not self.face_db_dir.exists():
            self.face_db_dir.mkdir(parents=True, exist_ok=True)

        people: dict[str, Any] = {}
        image_count: dict[str, int] = {}

        for person_dir in sorted(p for p in self.face_db_dir.iterdir() if p.is_dir()):
            embeddings = []
            image_paths = [
                p for p in person_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ]
            for image_path in image_paths:
                img = self.cv2.imread(str(image_path))
                if img is None:
                    log.warning("skip unreadable image path=%s", image_path)
                    continue
                faces = self.app.get(img)
                if not faces:
                    log.warning("skip no-face image path=%s", image_path)
                    continue
                face = self._largest_face(faces)
                embeddings.append(face.normed_embedding)

            if embeddings:
                mean_embedding = self.np.mean(self.np.asarray(embeddings), axis=0)
                mean_embedding = mean_embedding / self.np.linalg.norm(mean_embedding)
                people[person_dir.name] = mean_embedding.astype(self.np.float32)
                image_count[person_dir.name] = len(embeddings)
            else:
                image_count[person_dir.name] = 0

        self.people = people
        self.image_count = image_count
        log.info("loaded people_count=%s face_db=%s", len(self.people), self.face_db_dir)

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "people_count": len(self.people)}

    def identify_base64(
        self,
        image_base64: str,
        image_filename: str | None = None,
        image_content_type: str | None = None,
    ) -> VisionProfile:
        _ = image_filename
        _ = image_content_type
        image_bytes = self._decode_base64_image(image_base64)
        if len(image_bytes) > self.settings.max_image_bytes:
            raise ValueError(f"Image too large: {len(image_bytes)} bytes")
        img = self._decode_image_bytes(image_bytes)
        faces = self.app.get(img)
        if not faces:
            return VisionProfile(
                recognized=False,
                confidence=0.0,
                source="none",
                face_count=0,
                reason="no_face_detected",
            )

        face = self._largest_face(faces)
        match_name, score = self._match_embedding(face.normed_embedding)
        age = self._extract_age(face)
        gender = self._extract_gender(face)
        bbox = [int(x) for x in face.bbox]

        if match_name and score >= self.settings.match_threshold:
            return VisionProfile(
                recognized=True,
                name=match_name,
                age=age,
                gender=gender,
                confidence=score,
                source="face_db",
                face_count=len(faces),
                bbox=bbox,
            )

        if gender is not None:
            return VisionProfile(
                recognized=False,
                name=None,
                age=age,
                gender=gender,
                confidence=max(0.0, score),
                source="local_face_analysis",
                face_count=len(faces),
                bbox=bbox,
                reason="unknown_face_local_gender_only",
            )

        vlm_gender = self._infer_gender_with_vlm(image_bytes)
        if vlm_gender is not None:
            return VisionProfile(
                recognized=False,
                name=None,
                age=age,
                gender=vlm_gender,
                confidence=max(0.0, score),
                source="vlm",
                face_count=len(faces),
                bbox=bbox,
                reason="unknown_face_vlm_gender",
            )

        return VisionProfile(
            recognized=False,
            name=None,
            age=age,
            gender=None,
            confidence=max(0.0, score),
            source="none",
            face_count=len(faces),
            bbox=bbox,
            reason="unknown_face_no_gender",
        )

    def _match_embedding(self, embedding: Any) -> tuple[str | None, float]:
        if not self.people:
            return None, 0.0

        best_name = None
        best_score = -1.0
        for name, known_embedding in self.people.items():
            score = float(self.np.dot(embedding, known_embedding))
            if score > best_score:
                best_name = name
                best_score = score
        return best_name, best_score

    @staticmethod
    def _largest_face(faces):
        def area(face) -> float:
            x1, y1, x2, y2 = face.bbox
            return float(max(0, x2 - x1) * max(0, y2 - y1))

        return max(faces, key=area)

    def _decode_image_bytes(self, image_bytes: bytes):
        img_array = self.np.frombuffer(image_bytes, self.np.uint8)
        img = self.cv2.imdecode(img_array, self.cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image")
        return img

    @staticmethod
    def _decode_base64_image(value: str) -> bytes:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("image_base64 is empty")
        if cleaned.startswith("data:"):
            _, _, cleaned = cleaned.partition(",")
        try:
            return base64.b64decode(cleaned, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Invalid base64 image payload") from exc

    @staticmethod
    def _extract_age(face) -> int | None:
        raw = getattr(face, "age", None)
        if raw is None:
            return None
        try:
            age = int(raw)
        except (TypeError, ValueError):
            return None
        if age < 0 or age > 120:
            return None
        return age

    @staticmethod
    def _extract_gender(face) -> str | None:
        raw = getattr(face, "sex", None)
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            if lowered in {"f", "female", "woman"}:
                return "female"
            if lowered in {"m", "male", "man"}:
                return "male"

        raw = getattr(face, "gender", None)
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            if lowered in {"female", "woman", "f"}:
                return "female"
            if lowered in {"male", "man", "m"}:
                return "male"
        if isinstance(raw, (int, float)):
            if int(raw) == 0:
                return "female"
            if int(raw) == 1:
                return "male"
        return None

    def _infer_gender_with_vlm(self, image_bytes: bytes) -> str | None:
        if not self.settings.vlm_enabled:
            return None
        if not self.settings.vlm_api_url or not self.settings.vlm_model:
            return None
        if self.settings.vlm_api_format != "openai":
            log.warning("unsupported vision vlm api format format=%s", self.settings.vlm_api_format)
            return None

        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self.settings.vlm_model,
            "temperature": 0.0,
            "max_tokens": 8,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You identify only apparent gender presentation for polite Vietnamese honorifics. "
                        "Return exactly one token: male, female, or unknown."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Return male, female, or unknown."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                },
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.settings.vlm_api_key:
            value = self.settings.vlm_api_key
            header = self.settings.vlm_api_key_header or "Authorization"
            if header.lower() == "authorization" and not value.lower().startswith("bearer "):
                value = f"Bearer {value}"
            headers[header] = value

        req = urllib.request.Request(
            self.settings.vlm_api_url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.settings.vlm_timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            log.warning("vlm http error code=%s detail=%s", exc.code, detail[:200])
            return None
        except urllib.error.URLError as exc:
            log.warning("vlm unreachable reason=%s", exc.reason)
            return None
        except socket.timeout:
            log.warning("vlm timeout after %ss", self.settings.vlm_timeout_sec)
            return None
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("vlm request failed error=%s", exc)
            return None

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("vlm returned invalid json")
            return None

        content = self._extract_openai_message_text(parsed)
        return self._normalize_gender_text(content)

    @staticmethod
    def _extract_openai_message_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                if text:
                    parts.append(text)
            return " ".join(parts).strip()
        return ""

    @staticmethod
    def _normalize_gender_text(text: str) -> str | None:
        lowered = re.sub(r"[^a-z]", " ", str(text or "").strip().lower())
        lowered = " ".join(lowered.split())
        if not lowered:
            return None
        if "female" in lowered or "woman" in lowered:
            return "female"
        if "male" in lowered or re.search(r"\bman\b", lowered):
            return "male"
        return None
