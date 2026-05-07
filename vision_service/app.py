from __future__ import annotations

import base64
import binascii
import hashlib
import logging

from fastapi import FastAPI
from fastapi import HTTPException

from vision_service.config import get_settings
from vision_service.schemas import HealthResponse
from vision_service.schemas import VisionIdentifyRequest
from vision_service.schemas import VisionProfile

log = logging.getLogger("vision-service")


def _request_image_meta(image_base64: str) -> tuple[int, int | None, str | None, str | None]:
    cleaned = str(image_base64 or "").strip()
    if cleaned.startswith("data:"):
        _, _, cleaned = cleaned.partition(",")
    b64_len = len(cleaned)
    if not cleaned:
        return b64_len, None, None, "empty"
    try:
        raw = base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError) as exc:
        return b64_len, None, None, str(exc)
    return b64_len, len(raw), hashlib.sha1(raw).hexdigest()[:16], None


def create_app(service=None) -> FastAPI:
    app = FastAPI(title="Noble Vision Service", version="0.1.0")
    settings = get_settings()
    holder = {"service": service}

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )

    def get_service_instance():
        if holder["service"] is None:
            from vision_service.service import VisionService

            holder["service"] = VisionService(settings)
        return holder["service"]

    @app.on_event("startup")
    def startup() -> None:
        svc = get_service_instance()
        log.info("startup host=%s port=%s people_count=%s", settings.host, settings.port, len(svc.people))

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        svc = get_service_instance()
        payload = HealthResponse(**svc.health())
        return payload

    @app.post("/vision/identify", response_model=VisionProfile)
    def identify(payload: VisionIdentifyRequest) -> VisionProfile:
        svc = get_service_instance()
        b64_len, byte_size, image_sha1, predecode_error = _request_image_meta(payload.image_base64)
        log.info(
            (
                "vision identify request: image_b64_len=%s image_bytes=%s image_sha1=%s predecode_error=%s "
                "filename=%s content_type=%s"
            ),
            b64_len,
            byte_size,
            image_sha1,
            predecode_error,
            payload.image_filename,
            payload.image_content_type,
        )
        try:
            return svc.identify_base64(
                image_base64=payload.image_base64,
                image_filename=payload.image_filename,
                image_content_type=payload.image_content_type,
            )
        except ValueError as exc:
            log.warning(
                "vision identify bad image: %s image_b64_len=%s image_bytes=%s image_sha1=%s",
                exc,
                b64_len,
                byte_size,
                image_sha1,
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            log.warning(
                "vision identify runtime error: %s image_b64_len=%s image_bytes=%s image_sha1=%s",
                exc,
                b64_len,
                byte_size,
                image_sha1,
            )
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


app = create_app()
