from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi import HTTPException

from vision_service.config import get_settings
from vision_service.schemas import HealthResponse
from vision_service.schemas import VisionIdentifyRequest
from vision_service.schemas import VisionProfile

log = logging.getLogger("vision-service")


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
        try:
            return svc.identify_base64(
                image_base64=payload.image_base64,
                image_filename=payload.image_filename,
                image_content_type=payload.image_content_type,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


app = create_app()
