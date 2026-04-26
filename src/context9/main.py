"""End-to-end documentation ingestion demo."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich import print as rich_print

from context9.config import get_settings
from context9.embeddings import HashEmbeddingService
from context9.fetcher import fetch_document
from context9.qdrant_store import QdrantDocumentStore
from context9.text import build_chunks, extract_html_text

if TYPE_CHECKING:
    from context9.models import SearchCandidate

DEFAULT_SOURCE = "https://fastapi.tiangolo.com/advanced/custom-response/#custom-response-class"
DEFAULT_PACKAGE = "fastapi"
DEFAULT_VERSION = "0.136.1"
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_QUERY = "What is default_response_class in FastAPI"


@dataclass(frozen=True)
class IngestionFlowResult:
    """Summary of the documentation ingestion flow.

    Attributes:
        collection: Qdrant collection used for storage.
        source: Documentation source URL.
        title: Extracted page title, if present.
        chunks: Number of chunks stored.
        query: Query embedded and used to search Qdrant.
        candidates: Candidates returned from Qdrant for the smoke query.
    """

    collection: str
    source: str
    query: str
    candidates: list[SearchCandidate]
    chunks: int = 0
    title: str | None = None


async def main(
    source: str = DEFAULT_SOURCE,
    *,
    package: str = DEFAULT_PACKAGE,
    version: str = DEFAULT_VERSION,
    search_limit: int = DEFAULT_SEARCH_LIMIT,
    query: str = DEFAULT_QUERY,
) -> IngestionFlowResult:
    """Fetch documentation, chunk it, store it in Qdrant, and query it back.

    Args:
        source: Documentation URL or local path to ingest.
        package: Package or project name associated with the documentation.
        version: Documentation version label.
        search_limit: Number of candidates to fetch from Qdrant for the smoke query.
        query: Search query to embed and use against Qdrant.

    Returns:
        Summary of the collection, chunks, and retrieved candidates.
    """
    settings = get_settings()
    document = await fetch_document(source)
    title, text = extract_html_text(document.body)
    chunks = build_chunks(
        package=package,
        version=version,
        source_url=source,
        title=title,
        text=text,
    )
    embedding_service = HashEmbeddingService(dimensions=settings.embedding_dimensions)
    vectors = embedding_service.embed_texts([chunk.text for chunk in chunks])
    query_vector = embedding_service.embed_texts([query])[0]
    store = QdrantDocumentStore(settings)
    collection = store.upsert_chunks(chunks, vectors)
    candidates = store.search(
        package=package,
        version=version,
        vector=query_vector,
        limit=search_limit,
    )
    return IngestionFlowResult(
        collection=collection,
        source=source,
        query=query,
        title=title,
        chunks=len(chunks),
        candidates=candidates,
    )


def run() -> None:
    """Run the ingestion flow and print a concise summary."""
    result = asyncio.run(main())
    rich_print(result)


if __name__ == "__main__":  # pragma: no cover
    run()
