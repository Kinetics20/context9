from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from context9.fetcher import RawDocument
from context9.main import DEFAULT_QUERY, IngestionFlowResult, main, run
from context9.models import SearchCandidate

if TYPE_CHECKING:
    from collections.abc import Sequence


class FakeEmbeddingService:
    instances: list[FakeEmbeddingService] = []

    def __init__(self, *, dimensions: int) -> None:
        """Record configured dimensions and calls for assertions."""
        self.dimensions = dimensions
        self.calls: list[list[str]] = []
        FakeEmbeddingService.instances.append(self)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text)), float(self.dimensions)] for text in texts]


class FakeStore:
    instances: list[FakeStore] = []

    def __init__(self, settings: object) -> None:
        """Record constructed stores for assertions."""
        self.settings = settings
        self.upserted_chunks: object = None
        self.upserted_vectors: object = None
        FakeStore.instances.append(self)

    def upsert_chunks(self, chunks: object, vectors: object) -> str:
        self.upserted_chunks = chunks
        self.upserted_vectors = vectors
        return "context9_fastapi_latest"

    def search(self, *, package: str, version: str, vector: Sequence[float], limit: int) -> list[SearchCandidate]:
        assert package == "fastapi"
        assert version == "latest"
        assert vector == [float(len(DEFAULT_QUERY)), 384.0]
        assert limit == 2
        return [
            SearchCandidate.model_validate(
                {
                    "id": "chunk-1",
                    "text": "FastAPI response docs.",
                    "score": 0.9,
                    "package": package,
                    "version": version,
                    "source_url": "https://docs.example.com/page",
                    "ordinal": 0,
                }
            )
        ]


@pytest.mark.asyncio
async def test_main_orchestrates_fetch_chunk_embed_store_and_search(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeEmbeddingService.instances = []
    FakeStore.instances = []

    async def fake_fetch_document(source: str) -> RawDocument:
        assert source == "https://docs.example.com/page"
        return RawDocument(
            source=source,
            body="<html><head><title>Docs</title></head><body><main><p>FastAPI response docs.</p></main></body></html>",
            content_type="text/html",
        )

    monkeypatch.setattr("context9.main.fetch_document", fake_fetch_document)
    monkeypatch.setattr("context9.main.HashEmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr("context9.main.QdrantDocumentStore", FakeStore)

    result = await main(
        source="https://docs.example.com/page",
        version="latest",
        search_limit=2,
    )

    assert result == IngestionFlowResult(
        collection="context9_fastapi_latest",
        source="https://docs.example.com/page",
        query=DEFAULT_QUERY,
        title="Docs",
        chunks=1,
        candidates=FakeStore.instances[0].search(
            package="fastapi",
            version="latest",
            vector=[float(len(DEFAULT_QUERY)), 384.0],
            limit=2,
        ),
    )
    assert FakeEmbeddingService.instances[0].calls == [["FastAPI response docs."], [DEFAULT_QUERY]]
    assert len(FakeStore.instances) == 1


def test_run_prints_main_result(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = IngestionFlowResult(collection="docs", source="source", query="query", candidates=[])

    async def fake_main() -> IngestionFlowResult:
        return expected

    printed: list[object] = []
    monkeypatch.setattr("context9.main.main", fake_main)
    monkeypatch.setattr("context9.main.rich_print", printed.append)

    run()

    assert printed == [expected]
