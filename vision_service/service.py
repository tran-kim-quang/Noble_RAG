from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import ValidationError

from vision_service.config import Settings
from vision_service.schemas import VisionProfile

log = logging.getLogger("vision-service")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PersonalInference(BaseModel):
    gender: str
    age: str


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
        self.temp_faces: dict[str, dict[str, Any]] = {}
        self._vlm_client = None
        self.load()

    def load(self) -> None:
        if not self.face_db_dir.exists():
            self.face_db_dir.mkdir(parents=True, exist_ok=True)

        people: dict[str, Any] = {}
        image_count: dict[str, int] = {}

        for person_dir in sorted(p for p in self.face_db_dir.iterdir() if p.is_dir()):
            embeddings = []
            image_paths = [p for p in person_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
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
        self._purge_expired_temp_faces()
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
        embedding = face.normed_embedding
        match_name, score = self._match_embedding(embedding)
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
                face_id=self._known_face_id(match_name),
            )

        temp_face_id, temp_score = self._get_or_create_temp_face_id(embedding, age=age, gender=gender)

        if gender is not None and age is not None:
            return VisionProfile(
                recognized=False,
                name=None,
                age=age,
                gender=gender,
                confidence=max(0.0, temp_score, score),
                source="local_face_analysis",
                face_count=len(faces),
                bbox=bbox,
                reason="unknown_face_local_gender_only",
                face_id=temp_face_id,
            )

        vlm_age, vlm_gender = self._infer_personal_with_vlm(image_bytes)
        merged_age = age or vlm_age
        merged_gender = gender or vlm_gender
        if merged_age is not None or merged_gender is not None:
            self._update_temp_face(temp_face_id, age=merged_age, gender=merged_gender)
            return VisionProfile(
                recognized=False,
                name=None,
                age=merged_age,
                gender=merged_gender,
                confidence=max(0.0, temp_score, score),
                source="vlm",
                face_count=len(faces),
                bbox=bbox,
                reason="unknown_face_vlm_personal",
                face_id=temp_face_id,
            )

        return VisionProfile(
            recognized=False,
            name=None,
            age=age,
            gender=None,
            confidence=max(0.0, temp_score, score),
            source="none",
            face_count=len(faces),
            bbox=bbox,
            reason="unknown_face_no_gender",
            face_id=temp_face_id,
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

    def _match_temp_embedding(self, embedding: Any) -> tuple[str | None, float]:
        self._purge_expired_temp_faces()
        best_id = None
        best_score = -1.0
        for temp_id, payload in self.temp_faces.items():
            known_embedding = payload.get("embedding")
            if known_embedding is None:
                continue
            score = float(self.np.dot(embedding, known_embedding))
            if score > best_score:
                best_id = temp_id
                best_score = score
        return best_id, best_score

    def _get_or_create_temp_face_id(
        self,
        embedding: Any,
        age: str | None,
        gender: str | None,
    ) -> tuple[str, float]:
        temp_face_id, score = self._match_temp_embedding(embedding)
        if temp_face_id and score >= self.settings.temp_face_match_threshold:
            self._touch_temp_face(temp_face_id, embedding=embedding, age=age, gender=gender)
            return temp_face_id, score

        digest = hashlib.sha1(self.np.asarray(embedding).astype(self.np.float32).tobytes()).hexdigest()[:16]
        temp_face_id = f"guest-{digest}"
        self.temp_faces[temp_face_id] = {
            "embedding": self._normalize_embedding(embedding),
            "age": age,
            "gender": gender,
            "expires_at": time.time() + max(1, self.settings.temp_face_ttl_sec),
        }
        return temp_face_id, max(0.0, score)

    def _touch_temp_face(
        self,
        temp_face_id: str,
        *,
        embedding: Any,
        age: str | None,
        gender: str | None,
    ) -> None:
        payload = self.temp_faces.get(temp_face_id)
        if payload is None:
            payload = {}
            self.temp_faces[temp_face_id] = payload
        payload["embedding"] = self._normalize_embedding(embedding)
        if age is not None:
            payload["age"] = age
        if gender is not None:
            payload["gender"] = gender
        payload["expires_at"] = time.time() + max(1, self.settings.temp_face_ttl_sec)

    def _update_temp_face(self, temp_face_id: str, age: str | None, gender: str | None) -> None:
        payload = self.temp_faces.get(temp_face_id)
        if payload is None:
            return
        if age is not None:
            payload["age"] = age
        if gender is not None:
            payload["gender"] = gender
        payload["expires_at"] = time.time() + max(1, self.settings.temp_face_ttl_sec)

    def _purge_expired_temp_faces(self) -> None:
        now = time.time()
        expired = [face_id for face_id, payload in self.temp_faces.items() if float(payload.get("expires_at", 0)) <= now]
        for face_id in expired:
            self.temp_faces.pop(face_id, None)

    def _normalize_embedding(self, embedding: Any):
        normalized = self.np.asarray(embedding).astype(self.np.float32)
        denom = self.np.linalg.norm(normalized)
        if denom == 0:
            return normalized
        return normalized / denom

    @staticmethod
    def _known_face_id(name: str) -> str:
        normalized = re.sub(r"\s+", "-", str(name or "").strip().lower())
        normalized = re.sub(r"[^a-z0-9_\-]", "", normalized)
        return normalized or "known-customer"

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
    def _extract_age(face) -> str | None:
        for attr_name in ("age_group", "age"):
            raw = getattr(face, attr_name, None)
            if raw is None:
                continue
            normalized = VisionService._normalize_age_enum(raw)
            if normalized is not None:
                return normalized
        return None

    @staticmethod
    def _normalize_age_enum(raw: Any) -> str | None:
        if isinstance(raw, (int, float)):
            # Backward compatibility: some face models still output numeric age.
            return "trẻ" if int(raw) < 40 else "trung niên"

        lowered = str(raw).strip().lower()
        if not lowered:
            return None
        if lowered in {"trẻ", "tre", "young"}:
            return "trẻ"
        if lowered in {"trung niên", "trung nien", "middle", "middle-aged", "middle_aged"}:
            return "trung niên"
        return None

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

    def _infer_personal_with_vlm(self, image_bytes: bytes) -> tuple[str | None, str | None]:
        if not self.settings.vlm_enabled:
            return None, None
        if not self.settings.vlm_api_url or not self.settings.vlm_model:
            return None, None
        if self.settings.vlm_api_format not in {"ollama", "openai"}:
            log.warning("unsupported vision vlm api format format=%s", self.settings.vlm_api_format)
            return None, None

        try:
            client = self._get_ollama_client()
            content = self._ollama_chat_personal(client, image_bytes=image_bytes)
        except RuntimeError as exc:
            log.warning("vlm unavailable error=%s", exc)
            return None, None
        except Exception as exc:  # pragma: no cover
            log.warning("vlm request failed error=%s", exc)
            return None, None

        try:
            payload = PersonalInference.model_validate_json(content)
        except ValidationError:
            return None, None

        age = self._normalize_age_enum(payload.age)
        gender = self._normalize_gender_text(payload.gender)
        return age, gender

    def _get_ollama_client(self):
        if self._vlm_client is not None:
            return self._vlm_client
        try:
            from ollama import Client
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("ollama package is not installed") from exc

        headers = {}
        if self.settings.vlm_api_key:
            value = self.settings.vlm_api_key
            header = self.settings.vlm_api_key_header or "Authorization"
            if header.lower() == "authorization" and not value.lower().startswith("bearer "):
                value = f"Bearer {value}"
            headers[header] = value

        self._vlm_client = Client(host=self.settings.vlm_api_url, headers=headers or None)
        return self._vlm_client

    def _ollama_chat_personal(self, client: Any, image_bytes: bytes) -> str:
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": (
                    "Phạm vi các thuộc tính:\n"
                    "<giới tính>: [nam, nữ]\n"
                    "<độ tuổi>: [trẻ, trung niên]\n\n"
                    "Phân tích bức ảnh và chỉ trả lời bằng JSON sau:\n"
                    "{\n"
                    '  "gender": "<giới tính>",\n'
                    '  "age": "<độ tuổi>"\n'
                    "}"
                ),
                "images": [image_b64],
            },
        ]
        options = {"temperature": 0, "num_predict": 64}
        chunks: list[str] = []
        for part in client.chat(
            self.settings.vlm_model,
            messages=messages,
            stream=True,
            options=options,
            format=PersonalInference.model_json_schema(),
        ):
            msg = part.get("message", {}) if isinstance(part, dict) else {}
            text = str(msg.get("content", "")).strip()
            if text:
                chunks.append(text)
        content = "".join(chunks).strip()
        content = re.sub(r"^```json\s*|```$", "", content, flags=re.IGNORECASE | re.MULTILINE).strip()
        return content

    @staticmethod
    def _normalize_gender_text(text: str) -> str | None:
        lowered = re.sub(r"[^a-z\u00C0-\u1EF9]", " ", str(text or "").strip().lower())
        lowered = " ".join(lowered.split())
        if not lowered:
            return None
        if lowered in {"nam"}:
            return "male"
        if lowered in {"nu", "nữ"}:
            return "female"
        if "female" in lowered or "woman" in lowered:
            return "female"
        if "male" in lowered or re.search(r"\bman\b", lowered):
            return "male"
        return None
