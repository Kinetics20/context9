"""Qdrant persistence for Context9 chunks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NoReturn, Protocol, cast

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from context9.models import DocumentChunk, Metadata, MetadataValue, SearchCandidate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import SecretStr

    from context9.config import Settings

_COLLECTION_SAFE_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_COLLECTION_UNDERSCORE_RE = re.compile(r"_+")
_DEFAULT_COLLECTION_NAME = "context9"


class QdrantStoreError(RuntimeError):
    """Base exception for Qdrant persistence failures."""


class QdrantAuthenticationError(QdrantStoreError):
    """Raised when Qdrant rejects configured credentials."""


class DocumentStore(Protocol):
    """Persistence boundary for document chunks."""

    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
    ) -> str:
        """Store chunks and vectors.

        Args:
            chunks: Chunks to store.
            vectors: Dense vectors aligned with chunks.

        Returns:
            Collection name.
        """

    def search(
        self,
        *,
        package: str,
        version: str,
        vector: Sequence[float],
        limit: int,
    ) -> list[SearchCandidate]:
        """Search for nearest chunks.

        Args:
            package: Package name.
            version: Package version.
            vector: Query vector.
            limit: Maximum candidate count.

        Returns:
            Candidate chunks.
        """


class QdrantDocumentStore:
    """DocumentStore backed by Qdrant collections."""

    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        """Initialize the Qdrant client.

        Args:
            settings: Application settings.
            client: Optional preconfigured Qdrant client, useful for tests.
        """
        self._settings = settings
        self._client = client or QdrantClient(
            url=str(settings.qdrant_url),
            api_key=_secret_value(settings.qdrant_api_key),
            timeout=settings.qdrant_timeout_seconds,
        )

    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
    ) -> str:
        """Store chunks in a package/version collection."""
        if len(chunks) != len(vectors):
            msg = "chunks and vectors must have the same length"
            raise ValueError(msg)
        if not chunks:
            msg = "at least one chunk is required"
            raise ValueError(msg)
        _validate_vectors(vectors)

        vector_size = len(vectors[0])
        collection = collection_name(
            self._settings.collection_prefix,
            chunks[0].package,
            chunks[0].version,
        )
        self.ensure_collection(collection, vector_size)
        self.delete_source(collection, str(chunks[0].source_url))

        points = [
            qmodels.PointStruct(
                id=chunk.id,
                vector=list(vector),
                payload=_chunk_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        try:
            self._client.upsert(collection_name=collection, points=points)
        except UnexpectedResponse as exc:
            _raise_qdrant_error(exc)
        return collection

    def delete_source(self, collection: str, source_url: str) -> None:
        """Delete existing chunks for one source URL.

        Args:
            collection: Collection name.
            source_url: Source URL or local path to replace.
        """
        try:
            self._client.delete(
                collection_name=collection,
                points_selector=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="source_url",
                            match=qmodels.MatchValue(value=source_url),
                        )
                    ]
                ),
            )
        except UnexpectedResponse as exc:
            _raise_qdrant_error(exc)

    def search(
        self,
        *,
        package: str,
        version: str,
        vector: Sequence[float],
        limit: int,
    ) -> list[SearchCandidate]:
        """Search Qdrant for nearest chunks."""
        if limit <= 0:
            msg = "limit must be greater than zero"
            raise ValueError(msg)
        if not vector:
            msg = "query vector must not be empty"
            raise ValueError(msg)

        collection = collection_name(self._settings.collection_prefix, package, version)
        if not self._collection_exists(collection):
            return []

        try:
            response = self._client.query_points(
                collection_name=collection,
                query=list(vector),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except UnexpectedResponse as exc:
            _raise_qdrant_error(exc)
        return [_candidate_from_point(point) for point in response.points]

    def ensure_collection(self, collection: str, vector_size: int) -> None:
        """Create a collection if it does not already exist.

        Args:
            collection: Collection name.
            vector_size: Dense vector size.
        """
        if vector_size <= 0:
            msg = "vector_size must be greater than zero"
            raise ValueError(msg)
        if self._collection_exists(collection):
            return
        try:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
            )
        except UnexpectedResponse as exc:
            _raise_qdrant_error(exc)

    def _collection_exists(self, collection: str) -> bool:
        try:
            return self._client.collection_exists(collection_name=collection)
        except UnexpectedResponse as exc:
            _raise_qdrant_error(exc)


def collection_name(prefix: str, package: str, version: str) -> str:
    """Build a Qdrant-safe collection name.

    Args:
        prefix: Collection prefix.
        package: Package name.
        version: Version label.

    Returns:
        Sanitized collection name.
    """
    raw = f"{prefix}_{package}_{version}".lower()
    safe_name = _COLLECTION_SAFE_RE.sub("_", raw)
    return _COLLECTION_UNDERSCORE_RE.sub("_", safe_name).strip("_") or _DEFAULT_COLLECTION_NAME


def _chunk_payload(chunk: DocumentChunk) -> dict[str, object]:
    """Build Qdrant payload for one chunk."""
    return {
        "package": chunk.package,
        "version": chunk.version,
        "source_url": str(chunk.source_url),
        "title": chunk.title,
        "text": chunk.text,
        "ordinal": chunk.ordinal,
        "metadata": chunk.metadata,
    }


def _candidate_from_point(point: qmodels.ScoredPoint) -> SearchCandidate:
    """Convert a Qdrant point into a search candidate."""
    payload = point.payload or {}
    metadata = payload.get("metadata")
    return SearchCandidate.model_validate(
        {
            "id": str(point.id),
            "text": _payload_str(payload, "text"),
            "score": float(point.score),
            "package": _payload_str(payload, "package"),
            "version": _payload_str(payload, "version"),
            "source_url": _payload_str(payload, "source_url"),
            "title": _payload_optional_str(payload, "title"),
            "ordinal": _payload_int(payload, "ordinal"),
            "metadata": _clean_metadata(metadata if isinstance(metadata, dict) else {}),
        }
    )


def _payload_str(payload: dict[str, object], key: str) -> str:
    """Read a string payload value."""
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _payload_optional_str(payload: dict[str, object], key: str) -> str | None:
    """Read an optional string payload value."""
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _payload_int(payload: dict[str, object], key: str) -> int:
    """Read an int payload value."""
    value = payload.get(key)
    return value if isinstance(value, int) else 0


def _clean_metadata(raw: dict[object, object]) -> Metadata:
    """Keep metadata values supported by the public metadata contract."""
    metadata: Metadata = {}
    for key, value in raw.items():
        clean_value = _clean_metadata_value(value)
        if isinstance(key, str) and clean_value is not None:
            metadata[key] = clean_value
    return metadata


def _clean_metadata_value(value: object) -> MetadataValue | None:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        cleaned_list = [_clean_metadata_value(item) for item in value]
        return cast("MetadataValue", [item for item in cleaned_list if item is not None])
    if isinstance(value, dict):
        return _clean_metadata(cast("dict[object, object]", value))
    return None


def _validate_vectors(vectors: Sequence[Sequence[float]]) -> None:
    if not vectors:
        msg = "at least one vector is required"
        raise ValueError(msg)
    expected_size = len(vectors[0])
    if expected_size <= 0:
        msg = "vectors must not be empty"
        raise ValueError(msg)
    if any(len(vector) != expected_size for vector in vectors):
        msg = "all vectors must have the same length"
        raise ValueError(msg)


def _secret_value(secret: SecretStr | None) -> str | None:
    return None if secret is None else secret.get_secret_value()


def _raise_qdrant_error(exc: UnexpectedResponse) -> NoReturn:
    if exc.status_code == 401:
        msg = (
            "Qdrant rejected the configured API key. Check QDRANT_API_KEY in .env and make sure it matches "
            "QDRANT__SERVICE__API_KEY used by the running Qdrant container."
        )
        raise QdrantAuthenticationError(msg) from exc
    msg = f"Qdrant request failed: {exc}"
    raise QdrantStoreError(msg) from exc
