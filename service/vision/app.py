import logging
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.routes_vision import router as vision_router
from core.config import get_settings
from stores.customer_identity_store import CustomerIdentityStore

settings = get_settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
log = logging.getLogger("vision-service")

app = FastAPI(
    title="Noble Vision Identity Service",
    description="Vision-based customer identification service for Noble agent stack.",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    await CustomerIdentityStore().ensure_customer_schema()
    log.info("Vision service started and customer schema ensured.")


app.include_router(vision_router)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.vision_service_host,
        port=settings.vision_service_port,
        log_level=settings.log_level.lower(),
    )
