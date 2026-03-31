from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[3] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Shared storage config from root .env
    postgres_url: str = ""

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "rag_user"
    postgres_password: str = "rag_password"
    postgres_db: str = "noble_rag"

    # Vision config
    vision_api_key: str = ""
    vision_base_url: str = ""
    vision_project_id: str = ""
    vision_max_tokens: int = 8000
    vision_model_name: str = "mvp-image-hash"
    vision_match_threshold: float = 0.75
    vision_embedding_dim: int = 512
    vision_min_face_size: int = 112
    vision_max_faces: int = 1
    vision_purchase_history_path: str = "./data/customer_purchase_history.json"

    vision_service_host: str = "0.0.0.0"
    vision_service_port: int = 8020
    log_level: str = "INFO"

    def model_post_init(self, __context) -> None:
        if not self.postgres_url:
            self.postgres_url = (
                f"postgresql://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
