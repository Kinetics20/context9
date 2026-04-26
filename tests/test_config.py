import pytest
from pydantic import ValidationError

from context9.config import Settings, get_settings


def test_settings_defaults_are_valid_without_env_file() -> None:
    settings = Settings(_env_file=None)

    assert str(settings.qdrant_url) == "http://localhost:6333/"
    assert settings.qdrant_api_key is None
    assert str(settings.search_api_url) == "http://localhost:8000/"
    assert settings.search_api_key is None
    assert str(settings.embedder_url) == "http://localhost:8500/"
    assert settings.embedder_api_key is None
    assert settings.embedding_dimensions == 384
    assert settings.chunk_max_chars == 1000
    assert settings.chunk_overlap == 160
    assert settings.collection_prefix == "context9"
    assert settings.qdrant_timeout_seconds == 10


def test_settings_load_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example.com")
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-secret")
    monkeypatch.setenv("SEARCH_API_URL", "https://search.example.com")
    monkeypatch.setenv("SEARCH_API_KEY", "search-secret")
    monkeypatch.setenv("EMBEDDER_URL", "https://embedder.example.com")
    monkeypatch.setenv("EMBEDDER_API_KEY", "embedder-secret")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.setenv("CHUNK_MAX_CHARS", "2000")
    monkeypatch.setenv("CHUNK_OVERLAP", "250")
    monkeypatch.setenv("COLLECTION_PREFIX", "docs")
    monkeypatch.setenv("QDRANT_TIMEOUT_SECONDS", "20")

    settings = Settings(_env_file=None)

    assert str(settings.qdrant_url) == "https://qdrant.example.com/"
    assert settings.qdrant_api_key is not None
    assert settings.qdrant_api_key.get_secret_value() == "qdrant-secret"
    assert str(settings.search_api_url) == "https://search.example.com/"
    assert settings.search_api_key is not None
    assert settings.search_api_key.get_secret_value() == "search-secret"
    assert str(settings.embedder_url) == "https://embedder.example.com/"
    assert settings.embedder_api_key is not None
    assert settings.embedder_api_key.get_secret_value() == "embedder-secret"
    assert settings.embedding_dimensions == 1024
    assert settings.chunk_max_chars == 2000
    assert settings.chunk_overlap == 250
    assert settings.collection_prefix == "docs"
    assert settings.qdrant_timeout_seconds == 20


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    [
        ("QDRANT_URL", "not a url"),
        ("SEARCH_API_URL", "not a url"),
        ("EMBEDDER_URL", "not a url"),
        ("EMBEDDING_DIMENSIONS", "7"),
        ("EMBEDDING_DIMENSIONS", "4097"),
        ("CHUNK_MAX_CHARS", "0"),
        ("CHUNK_OVERLAP", "-1"),
        ("QDRANT_TIMEOUT_SECONDS", "0"),
    ],
)
def test_settings_reject_invalid_environment_values(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    env_value: str,
) -> None:
    monkeypatch.setenv(env_name, env_value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_reject_chunk_overlap_greater_than_or_equal_to_max_chars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNK_MAX_CHARS", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")

    with pytest.raises(ValidationError, match="chunk_overlap must be less than chunk_max_chars"):
        Settings(_env_file=None)


def test_settings_ignore_empty_environment_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "")

    settings = Settings(_env_file=None)

    assert str(settings.qdrant_url) == "http://localhost:6333/"


def test_settings_parse_null_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_API_KEY", "null")

    settings = Settings(_env_file=None)

    assert settings.qdrant_api_key is None


def test_get_settings_returns_cached_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "512")

    first = get_settings()
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")
    second = get_settings()

    assert first is second
    assert second.embedding_dimensions == 512
    get_settings.cache_clear()
