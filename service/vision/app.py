from fastapi import FastAPI

from service.vision.api.routes_vision import router as vision_router


app = FastAPI(
    title="Noble Vision Service",
    version="1.0.0",
    description="Vision gateway for remote machines to submit prompts.",
)

app.include_router(vision_router, prefix="/api/vision", tags=["vision"])


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
