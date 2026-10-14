"""Connector settings, read from environment variables (or .env locally).

Deliberately separate from core/config.py: the connector is an outside client of the
core and shares no code with it.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Where the core service is, and the API key it issued to this store.
    core_api_url: str
    core_api_key: str


@lru_cache
def get_settings() -> Settings:
    return Settings()
