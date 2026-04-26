from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from context9.config import Settings, get_settings
from context9.embedder_api import create_app, get_embedding_service, run

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pytest


class FakeEmbeddingService:
    dimensions = 3

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(index + 1), float(len(text)), 0.5] for index, text in enumerate(texts)]


def test_healthz_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_embed_returns_vectors_from_configured_service() -> None:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, embedder_api_key="secret")
    app.dependency_overrides[get_embedding_service] = lambda: FakeEmbeddingService()
    client = TestClient(app)

    response = client.post("/embed", headers={"api-key": "secret"}, json={"texts": ["a", "abcd"]})

    assert response.status_code == 200
    assert response.json() == {
        "dimensions": 3,
        "vectors": [[1.0, 1.0, 0.5], [2.0, 4.0, 0.5]],
    }


def test_embed_rejects_missing_api_key_when_configured() -> None:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, embedder_api_key="secret")
    app.dependency_overrides[get_embedding_service] = lambda: FakeEmbeddingService()
    client = TestClient(app)

    response = client.post("/embed", json={"texts": ["a"]})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API key"}


def test_embed_rejects_invalid_payload() -> None:
    client = TestClient(create_app())

    response = client.post("/embed", json={"texts": []})

    assert response.status_code == 422


def test_rerank_returns_candidates_sorted_by_blended_score() -> None:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    client = TestClient(app)
    payload = {
        "query": "default response class",
        "candidates": [
            {
                "id": "vector",
                "text": "Unrelated candidate",
                "score": 0.5,
                "package": "fastapi",
                "version": "latest",
                "source_url": "https://fastapi.tiangolo.com/docs",
                "ordinal": 0,
            },
            {
                "id": "lexical",
                "text": "default response class",
                "score": 0.2,
                "package": "fastapi",
                "version": "latest",
                "source_url": "https://fastapi.tiangolo.com/docs",
                "ordinal": 1,
            },
        ],
    }

    response = client.post("/rerank", json=payload)

    assert response.status_code == 200
    assert [candidate["id"] for candidate in response.json()["results"]] == ["lexical", "vector"]


def test_run_starts_uvicorn_with_embedder_app(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    expected_host = ".".join(["0", "0", "0", "0"])

    def fake_run(app_path: str, *, host: str, port: int) -> None:
        calls.append({"app_path": app_path, "host": host, "port": port})

    monkeypatch.setattr("context9.embedder_api.uvicorn.run", fake_run)

    run()

    assert calls == [{"app_path": "context9.embedder_api:app", "host": expected_host, "port": 8500}]
