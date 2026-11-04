import pytest
from pydantic import ValidationError

from core.config import Settings


def test_settings_read_from_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@db:5432/x")
    monkeypatch.setenv("CORE_ADMIN_KEY", "admin-key")

    # _env_file=None: ignore any local .env, so the test sees only what it set.
    settings = Settings(_env_file=None)

    assert settings.database_url == "postgresql+psycopg://u:p@db:5432/x"
    assert settings.core_admin_key == "admin-key"


def test_missing_admin_key_fails_clearly(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@db:5432/x")
    monkeypatch.delenv("CORE_ADMIN_KEY", raising=False)

    with pytest.raises(ValidationError, match="core_admin_key"):
        Settings(_env_file=None)


def test_missing_database_url_fails_clearly(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("CORE_ADMIN_KEY", "admin-key")

    with pytest.raises(ValidationError, match="database_url"):
        Settings(_env_file=None)
