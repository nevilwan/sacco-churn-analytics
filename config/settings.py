"""
config/settings.py
──────────────────
Centralised settings loaded from environment variables.
Never import os.environ directly elsewhere — always use `get_settings()`.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Database ──────────────────────────────────────────────────
    database_url: str
    database_test_url: str = ""

    # ── Security ─────────────────────────────────────────────────
    tokenization_secret_key: str   # Fernet key for PII field encryption
    hmac_secret_key: str           # HMAC key for deterministic tokenization
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480

    # ── Application ──────────────────────────────────────────────
    app_env: str = "development"
    app_name: str = "SACCO Retention Analysis System"
    log_level: str = "INFO"
    min_cohort_display_size: int = 30   # Privacy: never show cohorts smaller than this

    # ── API ──────────────────────────────────────────────────────
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance — call this everywhere."""
    return Settings()
