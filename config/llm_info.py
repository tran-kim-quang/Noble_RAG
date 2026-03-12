"""
LLM Configuration using Pydantic Settings
Automatically loads configuration from environment variables
"""

from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator


class LLMConfig(BaseSettings):
    """
    LLM Configuration
    Tự động load từ environment variables hoặc .env file
    """
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )
    
    # General LLM Settings
    llm_api_key: str = Field(
        ...,
        description="API Key cho LLM provider"
    )
    
    llm_api_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL cho LLM API"
    )
    
    llm_model_name: str = Field(
        default="gpt-4o-mini",
        description="Tên model LLM"
    )
    
    llm_provider: str = Field(
        default="openai",
        description="LLM provider (openai, azure, ollama, gemini, huggingface)"
    )
    
    # OpenAI Specific
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API Key"
    )
    
    openai_base_url: Optional[str] = Field(
        default="https://api.openai.com/v1",
        description="OpenAI Base URL"
    )
    
    # Azure OpenAI Specific
    azure_openai_api_key: Optional[str] = Field(
        default=None,
        description="Azure OpenAI API Key"
    )
    
    azure_openai_endpoint: Optional[str] = Field(
        default=None,
        description="Azure OpenAI Endpoint"
    )
    
    azure_openai_api_version: Optional[str] = Field(
        default="2024-02-15-preview",
        description="Azure OpenAI API Version"
    )
    
    azure_openai_deployment_name: Optional[str] = Field(
        default=None,
        description="Azure OpenAI Deployment Name"
    )
    
    azure_openai_embedding_deployment_name: Optional[str] = Field(
        default=None,
        description="Azure OpenAI Embedding Deployment Name"
    )
    
    # Google Gemini Specific
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API Key"
    )
    
    # Ollama Specific
    ollama_host: Optional[str] = Field(
        default="http://localhost:11434",
        description="Ollama Host URL"
    )
    
    # Embedding Settings
    embedding_model_name: Optional[str] = Field(
        default=None,
        description="Tên model embedding"
    )
    
    embedding_dim: Optional[int] = Field(
        default=None,
        description="Số chiều của embedding vector"
    )
    
    max_token_size: int = Field(
        default=8192,
        description="Kích thước token tối đa"
    )
    
    @validator('llm_provider')
    def validate_provider(cls, v):
        """Validate provider name"""
        allowed_providers = ['openai', 'azure', 'ollama', 'gemini', 'huggingface']
        if v.lower() not in allowed_providers:
            raise ValueError(f"Provider must be one of {allowed_providers}")
        return v.lower()
    
    def get_api_key(self) -> str:
        """Get API key dựa trên provider"""
        provider_key_map = {
            'openai': self.openai_api_key or self.llm_api_key,
            'azure': self.azure_openai_api_key or self.llm_api_key,
            'gemini': self.gemini_api_key or self.llm_api_key,
            'ollama': None,  # Ollama không cần API key
            'huggingface': None  # HF có thể không cần API key
        }
        return provider_key_map.get(self.llm_provider, self.llm_api_key)
    
    def get_base_url(self) -> Optional[str]:
        """Get base URL dựa trên provider"""
        provider_url_map = {
            'openai': self.openai_base_url or self.llm_api_url,
            'azure': self.azure_openai_endpoint,
            'ollama': self.ollama_host,
            'gemini': None,
            'huggingface': None
        }
        return provider_url_map.get(self.llm_provider, self.llm_api_url)


class RAGConfig(BaseSettings):
    """
    RAG Configuration
    
    Storage Architecture:
    - PostgreSQL: Raw data (chunks, entities, relations, doc status)
    - Qdrant: Vector embeddings (optimized for similarity search)
    - Redis: Session/conversation cache (fast in-memory store)
    """
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )
    
    # RAG Working Directory
    rag_working_dir: str = Field(
        default="./rag_storage",
        description="Thư mục fallback cho local data (nếu cần)"
    )
    
    rag_workspace: Optional[str] = Field(
        default="default",
        description="Workspace name cho data isolation giữa các instances"
    )
    
    # ========== Storage - Raw Data (PostgreSQL) ==========
    kv_storage: str = Field(
        default="PGKVStorage",
        description="KV Storage type cho chunks, entities, relations"
    )
    
    postgres_host: str = Field(
        default="localhost",
        description="PostgreSQL host"
    )
    
    postgres_port: int = Field(
        default=5432,
        description="PostgreSQL port"
    )
    
    postgres_user: str = Field(
        default="rag_user",
        description="PostgreSQL username"
    )
    
    postgres_password: str = Field(
        default="",
        description="PostgreSQL password"
    )
    
    postgres_db: str = Field(
        default="noble_rag",
        description="PostgreSQL database name"
    )
    
    # ========== Storage - Vector Data (Qdrant) ==========
    vector_storage: str = Field(
        default="QdrantVectorDBStorage",
        description="Vector Storage type untuk embeddings"
    )
    
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL"
    )
    
    qdrant_collection: Optional[str] = Field(
        default=None,
        description="Qdrant collection name (auto-generated if None)"
    )
    
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key (nếu server require auth)"
    )
    
    # ========== Storage - Graph & Doc Status (PostgreSQL) ==========
    graph_storage: str = Field(
        default="PGGraphStorage",
        description="Graph Storage type cho knowledge graph"
    )
    
    doc_status_storage: str = Field(
        default="PGDocStatusStorage",
        description="Document Status Storage type"
    )
    
    # ========== Cache - Conversation Sessions (Redis) ==========
    redis_host: str = Field(
        default="localhost",
        description="Redis host"
    )
    
    redis_port: int = Field(
        default=6379,
        description="Redis port"
    )
    
    redis_db: int = Field(
        default=0,
        description="Redis database number"
    )
    
    redis_password: Optional[str] = Field(
        default=None,
        description="Redis password (nếu server require auth)"
    )
    
    redis_session_ttl: int = Field(
        default=86400,  # 1 day
        description="Redis session TTL in seconds (auto-expire)"
    )
    
    # ========== Knowledge Graph - MongoDB Alternative ==========
    use_mongodb_for_graph: bool = Field(
        default=False,
        description="Use MongoDB for knowledge graph instead of PostgreSQL. Recommendation: keep False, use PostgreSQL"
    )
    
    mongodb_host: str = Field(
        default="localhost",
        description="MongoDB host (only used if use_mongodb_for_graph=True)"
    )
    
    mongodb_port: int = Field(
        default=27017,
        description="MongoDB port"
    )
    
    mongodb_user: Optional[str] = Field(
        default=None,
        description="MongoDB username"
    )
    
    mongodb_password: Optional[str] = Field(
        default=None,
        description="MongoDB password"
    )
    
    mongodb_db: str = Field(
        default="noble_rag",
        description="MongoDB database name"
    )
    
    # ========== Chunk Settings ==========
    chunk_token_size: int = Field(
        default=1200,
        description="Token size mỗi chunk"
    )
    
    chunk_overlap_token_size: int = Field(
        default=100,
        description="Overlap token size giữa các chunks"
    )
    
    # ========== Query Settings ==========
    top_k: int = Field(
        default=60,
        description="Số lượng top results"
    )
    
    max_async: int = Field(
        default=4,
        description="Số lượng async processes tối đa"
    )
    
    def get_postgres_url(self) -> str:
        """Generate PostgreSQL connection URL"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    def get_redis_url(self) -> str:
        """Generate Redis connection URL"""
        if self.redis_password:
            return (
                f"redis://:{self.redis_password}"
                f"@{self.redis_host}:{self.redis_port}/{self.redis_db}"
            )
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    def get_mongodb_url(self) -> str:
        """Generate MongoDB connection URL (only if use_mongodb_for_graph=True)"""
        if self.mongodb_user and self.mongodb_password:
            return (
                f"mongodb://{self.mongodb_user}:{self.mongodb_password}"
                f"@{self.mongodb_host}:{self.mongodb_port}/{self.mongodb_db}"
            )
        return f"mongodb://{self.mongodb_host}:{self.mongodb_port}/{self.mongodb_db}"
    
    def get_qdrant_url(self) -> str:
        """Generate Qdrant URL"""
        return self.qdrant_url


class AppConfig(BaseSettings):
    """
    Application Configuration
    Combines all configs
    """
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )
    
    # App Settings
    app_name: str = Field(
        default="Noble_RAG",
        description="Application name"
    )
    
    app_version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    
    debug: bool = Field(
        default=False,
        description="Debug mode"
    )
    
    # Whisper Service Settings
    whisper_host: str = Field(
        default="localhost",
        description="Whisper service host"
    )
    
    whisper_port: int = Field(
        default=8000,
        description="Whisper service port"
    )


# Singleton instances
# ─────────────────────────────────────────────────────────
# Dùng @lru_cache(maxsize=None) thay vì global + manual None-check
#
# Tại sao lru_cache tốt hơn cách dùng global?
#
# 1. Thread-safe: CPython's GIL đảm bảo lru_cache chỉ gọi function
#    đúng 1 lần duy nhất dù có nhiều thread cùng gọi đồng thời.
#
# 2. Gọn hơn: Không cần khai báo `global _llm_config` thủ công.
#
# 3. Dễ test/reset: Gọi get_llm_config.cache_clear() để xóa cache,
#    lần gọi tiếp theo sẽ tạo instance mới từ .env hiện tại.
#
# Cơ chế hoạt động của lru_cache:
#   - Lần đầu gọi get_llm_config() → chạy thân hàm, cache kết quả
#   - Các lần tiếp theo → bỏ qua thân hàm, trả về kết quả đã cache
#   - maxsize=None → không giới hạn kích thước cache (luôn giữ 1 entry)
# ─────────────────────────────────────────────────────────


@lru_cache(maxsize=None)
def get_llm_config() -> LLMConfig:
    """
    Trả về LLMConfig singleton.
    Instance được tạo duy nhất 1 lần khi lần đầu gọi hàm,
    các lần tiếp theo trả về object đã cache — không đọc lại .env.

    Reset cache (dùng trong test):
        get_llm_config.cache_clear()
    """
    return LLMConfig()


@lru_cache(maxsize=None)
def get_rag_config() -> RAGConfig:
    """
    Trả về RAGConfig singleton.
    Instance được tạo duy nhất 1 lần khi lần đầu gọi hàm,
    các lần tiếp theo trả về object đã cache — không đọc lại .env.

    Reset cache (dùng trong test):
        get_rag_config.cache_clear()
    """
    return RAGConfig()


@lru_cache(maxsize=None)
def get_app_config() -> AppConfig:
    """
    Trả về AppConfig singleton.
    Instance được tạo duy nhất 1 lần khi lần đầu gọi hàm,
    các lần tiếp theo trả về object đã cache — không đọc lại .env.

    Reset cache (dùng trong test):
        get_app_config.cache_clear()
    """
    return AppConfig()


# For backward compatibility
def load_config():
    """Load all configurations"""
    llm = get_llm_config()
    rag = get_rag_config()
    app = get_app_config()
    return {
        'llm': llm,
        'rag': rag,
        'app': app
    }


if __name__ == "__main__":
    # Test configuration loading
    print("=== Testing Configuration Loading ===\n")
    
    # Load configs
    llm_config = get_llm_config()
    rag_config = get_rag_config()
    app_config = get_app_config()
    
    # Print LLM Config
    print("LLM Configuration:")
    print(f"  Provider: {llm_config.llm_provider}")
    print(f"  Model: {llm_config.llm_model_name}")
    print(f"  API URL: {llm_config.llm_api_url}")
    print(f"  API Key: {llm_config.llm_api_key[:10]}..." if llm_config.llm_api_key else "  API Key: None")
    print()
    
    # Print RAG Config
    print("RAG Configuration:")
    print(f"  Working Dir: {rag_config.rag_working_dir}")
    print(f"  Vector Storage: {rag_config.vector_storage}")
    print(f"  Graph Storage: {rag_config.graph_storage}")
    print(f"  Chunk Size: {rag_config.chunk_token_size}")
    print()
    
    # Print App Config
    print("App Configuration:")
    print(f"  App Name: {app_config.app_name}")
    print(f"  Version: {app_config.app_version}")
    print(f"  Debug: {app_config.debug}")
    print()
