import pytest
from pydantic import ValidationError

from core.config import Settings


def test_settings_read_from_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@db:5432/x")
    monkeypatch.setenv("CORE_API_KEYS", "key-a,key-b")

    # _env_file=None: ignore any local .env, so the test sees only what it set.
    settings = Settings(_env_file=None)

    assert settings.database_url == "postgresql+psycopg://u:p@db:5432/x"
    assert settings.core_api_keys == "key-a,key-b"


def test_missing_database_url_fails_clearly(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("CORE_API_KEYS", "key-a")

    with pytest.raises(ValidationError, match="database_url"):
        Settings(_env_file=None)
