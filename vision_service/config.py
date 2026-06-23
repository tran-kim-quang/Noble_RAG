from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("VISION_HOST", "0.0.0.0")
    port: int = int(os.getenv("VISION_PORT", "8031"))
    mock_enabled: bool = _env_bool("VISION_MOCK_ENABLED", "false")
    mock_recognized: bool = _env_bool("VISION_MOCK_RECOGNIZED", "true")
    mock_name: str = os.getenv("VISION_MOCK_NAME", "Linh").strip()
    mock_age: int = int(os.getenv("VISION_MOCK_AGE", "31"))
    mock_gender: str = os.getenv("VISION_MOCK_GENDER", "female").strip().lower()
    mock_confidence: float = float(os.getenv("VISION_MOCK_CONFIDENCE", "0.93"))
    mock_face_id: str = os.getenv("VISION_MOCK_FACE_ID", "known-linh").strip()
    mock_source: str = os.getenv("VISION_MOCK_SOURCE", "face_db").strip().lower()
    mock_face_count: int = int(os.getenv("VISION_MOCK_FACE_COUNT", "1"))

    face_db_dir: str = os.getenv("VISION_FACE_DB", "face_db").strip()
    model_name: str = os.getenv("VISION_MODEL_NAME", "buffalo_l").strip()
    match_threshold: float = float(os.getenv("VISION_MATCH_THRESHOLD", "0.45"))
    det_size: int = int(os.getenv("VISION_DET_SIZE", "640"))
    use_gpu: bool = _env_bool("VISION_USE_GPU", "true")

    vlm_enabled: bool = _env_bool("VISION_VLM_ENABLED", "false")
    vlm_api_format: str = os.getenv("VISION_VLM_API_FORMAT", "ollama").strip().lower()
    vlm_api_url: str = os.getenv("VISION_VLM_API_URL", "").strip()
    vlm_api_key: str = os.getenv("VISION_VLM_API_KEY", "").strip()
    vlm_api_key_header: str = os.getenv("VISION_VLM_API_KEY_HEADER", "Authorization").strip()
    vlm_model: str = os.getenv("VISION_VLM_MODEL", "").strip()
    vlm_timeout_sec: float = float(os.getenv("VISION_VLM_TIMEOUT_SEC", "20"))

    max_image_bytes: int = int(os.getenv("VISION_MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))
    temp_face_ttl_sec: int = int(os.getenv("VISION_TEMP_FACE_TTL_SEC", "300"))
    temp_face_match_threshold: float = float(os.getenv("VISION_TEMP_FACE_MATCH_THRESHOLD", "0.6"))
    shared_identity_db_path: str = os.getenv(
        "SHARED_IDENTITY_DB_PATH",
        "data/shared_identity/identity.sqlite3",
    ).strip()


def get_settings() -> Settings:
    return Settings()
