from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
import pytest
from pydantic import SecretStr
from qdrant_client import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from context9.config import Settings
from context9.models import DocumentChunk
from context9.qdrant_store import (
    QdrantAuthenticationError,
    QdrantDocumentStore,
    QdrantStoreError,
    _secret_value,
    _validate_vectors,
    collection_name,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def document_chunk_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "chunk-1",
        "package": "Fast API",
        "version": "0.136.1",
        "source_url": "https://fastapi.tiangolo.com/docs",
        "title": "FastAPI Docs",
        "text": "FastAPI response docs.",
        "ordinal": 0,
        "metadata": {"content_sha256": "abc"},
    }
    payload.update(overrides)
    return payload


@dataclass
class QueryResult:
    points: list[qmodels.ScoredPoint]


class FakeQdrantClient:
    def __init__(
        self,
        *,
        exists: bool = False,
        fail_status: int | None = None,
        fail_on: str | None = None,
    ) -> None:
        """Record Qdrant calls while optionally simulating failures."""
        self.exists = exists
        self.fail_status = fail_status
        self.fail_on = fail_on
        self.created: list[tuple[str, qmodels.VectorParams]] = []
        self.deleted: list[tuple[str, qmodels.Filter]] = []
        self.upserted: list[tuple[str, Sequence[qmodels.PointStruct]]] = []
        self.queries: list[dict[str, object]] = []
        self.query_points_result = QueryResult(points=[])

    def collection_exists(self, *, collection_name: str) -> bool:
        if self.fail_status is not None and self.fail_on in {None, "collection_exists"}:
            raise unexpected_response(self.fail_status)
        return self.exists or any(name == collection_name for name, _config in self.created)

    def create_collection(self, *, collection_name: str, vectors_config: qmodels.VectorParams) -> None:
        if self.fail_status is not None and self.fail_on in {None, "create"}:
            raise unexpected_response(self.fail_status)
        self.created.append((collection_name, vectors_config))

    def delete(self, *, collection_name: str, points_selector: qmodels.Filter) -> None:
        if self.fail_status is not None and self.fail_on in {None, "delete"}:
            raise unexpected_response(self.fail_status)
        self.deleted.append((collection_name, points_selector))

    def upsert(self, *, collection_name: str, points: Sequence[qmodels.PointStruct]) -> None:
        if self.fail_status is not None and self.fail_on in {None, "upsert"}:
            raise unexpected_response(self.fail_status)
        self.upserted.append((collection_name, points))

    def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ) -> QueryResult:
        if self.fail_status is not None and self.fail_on in {None, "query"}:
            raise unexpected_response(self.fail_status)
        self.queries.append(
            {
                "collection_name": collection_name,
                "query": query,
                "limit": limit,
                "with_payload": with_payload,
                "with_vectors": with_vectors,
            }
        )
        return self.query_points_result


def settings() -> Settings:
    return Settings(_env_file=None, collection_prefix="Docs")


def unexpected_response(status_code: int) -> UnexpectedResponse:
    return UnexpectedResponse(status_code, "Boom", b"{}", httpx.Headers())


def test_collection_name_sanitizes_parts_and_has_fallback() -> None:
    assert collection_name("Docs", "Fast API!", "0.136.1") == "docs_fast_api_0_136_1"
    assert collection_name("!!!", "***", "///") == "context9"


def test_upsert_chunks_creates_collection_deletes_existing_source_and_upserts_payloads() -> None:
    client = FakeQdrantClient()
    store = QdrantDocumentStore(settings(), client=client)  # type: ignore[arg-type]
    chunk = DocumentChunk.model_validate(document_chunk_payload(metadata={"keep": ["ok", {"nested": True}]}))

    collection = store.upsert_chunks([chunk], [[0.1, 0.2, 0.3]])

    assert collection == "docs_fast_api_0_136_1"
    assert client.created[0][0] == collection
    assert client.created[0][1].size == 3
    assert client.deleted[0][0] == collection
    assert client.upserted[0][0] == collection
    point = client.upserted[0][1][0]
    assert point.id == "chunk-1"
    assert point.vector == [0.1, 0.2, 0.3]
    assert point.payload == {
        "package": "Fast API",
        "version": "0.136.1",
        "source_url": "https://fastapi.tiangolo.com/docs",
        "title": "FastAPI Docs",
        "text": "FastAPI response docs.",
        "ordinal": 0,
        "metadata": {"keep": ["ok", {"nested": True}]},
    }


@pytest.mark.parametrize(
    ("chunks", "vectors", "match"),
    [
        ([], [], "at least one chunk is required"),
        ([DocumentChunk.model_validate(document_chunk_payload())], [], "chunks and vectors must have the same length"),
        (
            [DocumentChunk.model_validate(document_chunk_payload())],
            [[]],
            "vectors must not be empty",
        ),
        (
            [
                DocumentChunk.model_validate(document_chunk_payload(id="chunk-1", ordinal=0)),
                DocumentChunk.model_validate(document_chunk_payload(id="chunk-2", ordinal=1)),
            ],
            [[0.1], [0.1, 0.2]],
            "all vectors must have the same length",
        ),
    ],
)
def test_upsert_chunks_validates_inputs(
    chunks: list[DocumentChunk],
    vectors: list[list[float]],
    match: str,
) -> None:
    store = QdrantDocumentStore(settings(), client=FakeQdrantClient())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=match):
        store.upsert_chunks(chunks, vectors)


def test_search_returns_empty_list_when_collection_is_missing() -> None:
    store = QdrantDocumentStore(settings(), client=FakeQdrantClient(exists=False))  # type: ignore[arg-type]

    assert store.search(package="fastapi", version="latest", vector=[0.1], limit=3) == []


def test_search_maps_scored_points_to_candidates_and_cleans_metadata() -> None:
    client = FakeQdrantClient(exists=True)
    client.query_points_result = QueryResult(
        points=[
            qmodels.ScoredPoint(
                id="chunk-1",
                version=1,
                score=0.75,
                payload={
                    "text": "FastAPI docs",
                    "package": "fastapi",
                    "version": "latest",
                    "source_url": "https://fastapi.tiangolo.com/docs",
                    "title": "Docs",
                    "ordinal": 2,
                    "metadata": {"keep": ["ok", object()], "nested": {"enabled": True}, 1: "drop-key"},
                },
            )
        ]
    )
    store = QdrantDocumentStore(settings(), client=client)  # type: ignore[arg-type]

    results = store.search(package="fastapi", version="latest", vector=[0.1, 0.2], limit=1)

    assert len(results) == 1
    assert results[0].id == "chunk-1"
    assert results[0].score == 0.75
    assert results[0].metadata == {"keep": ["ok"], "nested": {"enabled": True}}
    assert client.queries == [
        {
            "collection_name": "docs_fastapi_latest",
            "query": [0.1, 0.2],
            "limit": 1,
            "with_payload": True,
            "with_vectors": False,
        }
    ]


def test_search_defaults_invalid_payload_values() -> None:
    client = FakeQdrantClient(exists=True)
    client.query_points_result = QueryResult(
        points=[
            qmodels.ScoredPoint(
                id="chunk-1",
                version=1,
                score=0.25,
                payload={
                    "text": "FastAPI docs",
                    "package": "fastapi",
                    "version": "latest",
                    "source_url": "https://fastapi.tiangolo.com/docs",
                    "title": 123,
                    "ordinal": "2",
                    "metadata": [],
                },
            )
        ]
    )
    store = QdrantDocumentStore(settings(), client=client)  # type: ignore[arg-type]

    result = store.search(package="fastapi", version="latest", vector=[0.1], limit=1)[0]

    assert result.text == "FastAPI docs"
    assert result.title is None
    assert result.ordinal == 0
    assert result.metadata == {}


def test_search_raises_validation_error_for_invalid_required_payload() -> None:
    client = FakeQdrantClient(exists=True)
    client.query_points_result = QueryResult(
        points=[
            qmodels.ScoredPoint(
                id="chunk-1",
                version=1,
                score=0.25,
                payload={
                    "text": 123,
                    "package": "fastapi",
                    "version": "latest",
                    "source_url": "https://fastapi.tiangolo.com/docs",
                    "ordinal": 0,
                },
            )
        ]
    )
    store = QdrantDocumentStore(settings(), client=client)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="String should have at least 1 character"):
        store.search(package="fastapi", version="latest", vector=[0.1], limit=1)


@pytest.mark.parametrize(
    ("vector", "limit", "match"),
    [
        ([0.1], 0, "limit must be greater than zero"),
        ([], 1, "query vector must not be empty"),
    ],
)
def test_search_validates_inputs(vector: list[float], limit: int, match: str) -> None:
    store = QdrantDocumentStore(settings(), client=FakeQdrantClient(exists=True))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=match):
        store.search(package="fastapi", version="latest", vector=vector, limit=limit)


def test_ensure_collection_rejects_invalid_vector_size() -> None:
    store = QdrantDocumentStore(settings(), client=FakeQdrantClient())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="vector_size must be greater than zero"):
        store.ensure_collection("docs", 0)


def test_ensure_collection_skips_create_when_collection_exists() -> None:
    client = FakeQdrantClient(exists=True)
    store = QdrantDocumentStore(settings(), client=client)  # type: ignore[arg-type]

    store.ensure_collection("docs", 3)

    assert client.created == []


def test_upsert_chunks_translates_upsert_errors() -> None:
    store = QdrantDocumentStore(
        settings(),
        client=FakeQdrantClient(fail_status=500, fail_on="upsert"),  # type: ignore[arg-type]
    )
    chunk = DocumentChunk.model_validate(document_chunk_payload())

    with pytest.raises(QdrantStoreError, match="Qdrant request failed"):
        store.upsert_chunks([chunk], [[0.1]])


def test_delete_source_translates_delete_errors() -> None:
    store = QdrantDocumentStore(
        settings(),
        client=FakeQdrantClient(fail_status=500, fail_on="delete"),  # type: ignore[arg-type]
    )

    with pytest.raises(QdrantStoreError, match="Qdrant request failed"):
        store.delete_source("docs", "https://fastapi.tiangolo.com/docs")


def test_search_translates_query_errors() -> None:
    store = QdrantDocumentStore(
        settings(),
        client=FakeQdrantClient(exists=True, fail_status=500, fail_on="query"),  # type: ignore[arg-type]
    )

    with pytest.raises(QdrantStoreError, match="Qdrant request failed"):
        store.search(package="fastapi", version="latest", vector=[0.1], limit=1)


def test_create_collection_translates_create_errors() -> None:
    store = QdrantDocumentStore(
        settings(),
        client=FakeQdrantClient(fail_status=500, fail_on="create"),  # type: ignore[arg-type]
    )

    with pytest.raises(QdrantStoreError, match="Qdrant request failed"):
        store.ensure_collection("docs", 3)


def test_validate_vectors_rejects_empty_vector_sequence() -> None:
    with pytest.raises(ValueError, match="at least one vector is required"):
        _validate_vectors([])


def test_secret_value_unwraps_secret_strings() -> None:
    assert _secret_value(None) is None
    assert _secret_value(SecretStr("secret")) == "secret"


def test_qdrant_401_is_raised_as_authentication_error() -> None:
    store = QdrantDocumentStore(settings(), client=FakeQdrantClient(fail_status=401))  # type: ignore[arg-type]

    with pytest.raises(QdrantAuthenticationError, match="Qdrant rejected"):
        store.ensure_collection("docs", 3)


def test_qdrant_unexpected_response_is_raised_as_store_error() -> None:
    store = QdrantDocumentStore(settings(), client=FakeQdrantClient(fail_status=500))  # type: ignore[arg-type]

    with pytest.raises(QdrantStoreError, match="Qdrant request failed"):
        store.ensure_collection("docs", 3)
