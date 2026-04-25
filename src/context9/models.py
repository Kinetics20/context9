from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

type MetadataPrimitive = str | int | float | bool | None
type MetadataValue = MetadataPrimitive | list[MetadataValue] | dict[str, MetadataValue]
type Metadata = dict[str, MetadataValue]
type NonEmptyStr = Annotated[str, Field(min_length=1)]
type NonEmptyStrList = Annotated[list[NonEmptyStr], Field(min_length=1)]
type Vector = Annotated[list[float], Field(min_length=1)]
type VectorList = Annotated[list[Vector], Field(min_length=1)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_min_length=1,
        str_strip_whitespace=True,
        validate_default=True,
    )


class DocumentChunk(_StrictModel):
    """Validated documentation chunk ready for indexing.

    Attributes:
        id: Stable chunk identifier.
        package: Package or project name that owns the documentation.
        version: Documentation version, defaulting to ``latest``.
        source_url: HTTP URL of the source document.
        title: Optional source document or section title.
        text: Chunk text content.
        ordinal: Zero-based chunk position within the source document.
        metadata: Additional structured metadata for filtering and retrieval.
    """

    id: NonEmptyStr = Field(description="Stable chunk identifier.")
    package: NonEmptyStr = Field(description="Package or project name that owns the documentation.")
    version: NonEmptyStr = Field(default="latest", description="Documentation version.")
    source_url: HttpUrl = Field(description="HTTP URL of the source document.")
    title: str | None = None
    text: NonEmptyStr = Field(description="Chunk text content.")
    ordinal: int = Field(ge=0, description="Zero-based chunk position within the source document.")
    metadata: Metadata = Field(default_factory=dict, description="Structured metadata for filtering and retrieval.")


class EmbedRequest(_StrictModel):
    """Request payload for creating embeddings.

    Attributes:
        texts: Non-empty text inputs to embed.
    """

    texts: NonEmptyStrList = Field(description="Non-empty text inputs to embed.")


class EmbedResponse(_StrictModel):
    """Embedding vectors returned by an embedding provider.

    Attributes:
        dimensions: Number of dimensions in each vector.
        vectors: Embedding vectors, one vector per input text.
    """

    dimensions: int = Field(ge=1, description="Number of dimensions in each vector.")
    vectors: VectorList = Field(description="Embedding vectors.")

    @model_validator(mode="after")
    def _validate_vector_dimensions(self) -> Self:
        expected_dimensions = self.dimensions
        if any(len(vector) != expected_dimensions for vector in self.vectors):
            msg = "All vectors must match the declared dimensions."
            raise ValueError(msg)
        return self


class SearchCandidate(_StrictModel):
    """Single candidate returned by search or reranking.

    Attributes:
        id: Stable chunk identifier.
        text: Candidate text content.
        score: Retrieval or reranking score.
        package: Package or project name that owns the documentation.
        version: Documentation version.
        source_url: HTTP URL of the source document.
        title: Optional source document or section title.
        ordinal: Zero-based chunk position within the source document.
        metadata: Structured metadata for filtering and retrieval.
    """

    id: NonEmptyStr = Field(description="Stable chunk identifier.")
    text: NonEmptyStr = Field(description="Candidate text content.")
    score: float = Field(allow_inf_nan=False, description="Retrieval or reranking score.")
    package: NonEmptyStr = Field(description="Package or project name that owns the documentation.")
    version: NonEmptyStr = Field(description="Documentation version.")
    source_url: HttpUrl = Field(description="HTTP URL of the source document.")
    title: str | None = None
    ordinal: int = Field(ge=0, description="Zero-based chunk position within the source document.")
    metadata: Metadata = Field(default_factory=dict, description="Structured metadata for filtering and retrieval.")


type CandidateList = Annotated[list[SearchCandidate], Field(min_length=1)]


class RerankRequest(_StrictModel):
    """Request payload for reranking search candidates.

    Attributes:
        query: Search query used for reranking.
        candidates: Non-empty candidate list to rerank.
    """

    query: NonEmptyStr = Field(description="Search query used for reranking.")
    candidates: CandidateList = Field(description="Non-empty candidate list to rerank.")


class RerankResponse(_StrictModel):
    """Response payload containing reranked candidates.

    Attributes:
        results: Non-empty reranked candidates.
    """

    results: CandidateList = Field(description="Non-empty reranked candidates.")


class SearchResponse(_StrictModel):
    """Search response payload.

    Attributes:
        package: Package or project name searched.
        version: Documentation version searched.
        query: Search query.
        results: Non-empty search results.
    """

    package: NonEmptyStr = Field(description="Package or project name searched.")
    version: NonEmptyStr = Field(description="Documentation version searched.")
    query: NonEmptyStr = Field(description="Search query.")
    results: CandidateList = Field(description="Non-empty search results.")


class IngestReport(_StrictModel):
    """Summary produced after ingesting documentation.

    Attributes:
        package: Package or project name ingested.
        version: Documentation version ingested.
        source: Source identifier or URL ingested.
        chunks: Number of chunks produced.
        collection: Vector collection name used for storage.
    """

    package: NonEmptyStr = Field(description="Package or project name ingested.")
    version: NonEmptyStr = Field(description="Documentation version ingested.")
    source: NonEmptyStr = Field(description="Source identifier or URL ingested.")
    chunks: int = Field(ge=0, description="Number of chunks produced.")
    collection: NonEmptyStr = Field(description="Vector collection name used for storage.")
