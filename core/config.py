"""Core service settings, read from environment variables or the repo-root .env file.

Variable names match .env.example (case-insensitive): DATABASE_URL -> database_url.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": .env also holds connector settings (SHOPIFY_..., CORE_API_URL)
    # that the core does not use.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    # Comma-separated during development; replaced by a stores table later.
    core_api_keys: str


@lru_cache
def get_settings() -> Settings:
    """Load settings once, on first use.

    A function rather than a module-level `settings = Settings()`, so that importing the
    core does not fail when no .env is present (e.g. in CI); only code that actually
    needs a setting does.
    """
    return Settings()
