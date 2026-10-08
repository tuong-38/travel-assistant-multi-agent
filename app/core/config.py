from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    log_level: str = "INFO"
    database_url: SecretStr
    litellm_base_url: str = "http://litellm:4000/v1"
    litellm_api_key: SecretStr
    default_model: str = "travel_general"
    travel_tools_mcp_url: str = "http://travel-tools-mcp:8001/mcp"
    mcp_timeout_seconds: float = 5.0
    redis_url: str = "redis://redis:6379/0"
    redis_stream: str = "travel:jobs"
    redis_consumer_group: str = "travel-workers"
    redis_claim_idle_ms: int = Field(default=300_000, gt=0)
    redis_heartbeat_seconds: float = Field(default=30.0, gt=0)
    job_max_delivery_attempts: int = Field(default=5, ge=1)
    job_status_ttl_seconds: int = Field(default=604_800, gt=0)
    worker_shutdown_grace_seconds: float = Field(default=30.0, gt=0)

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
