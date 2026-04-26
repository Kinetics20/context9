from functools import lru_cache
from typing import Annotated, Self

from pydantic import AliasChoices, Field, HttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

type EmbeddingDimensions = Annotated[int, Field(ge=8, le=4096)]
type ChunkMaxChars = Annotated[int, Field(ge=1)]
type ChunkOverlap = Annotated[int, Field(ge=0)]

DEFAULT_QDRANT_URL = HttpUrl("http://localhost:6333")
DEFAULT_SEARCH_API_URL = HttpUrl("http://localhost:8000")
DEFAULT_EMBEDDER_URL = HttpUrl("http://localhost:8500")
DEFAULT_CHUNK_MAX_CHARS = 1000
DEFAULT_CHUNK_OVERLAP = 160


class Settings(BaseSettings):
    """Application settings loaded from environment variables and ``.env``.

    Attributes:
        qdrant_url: Base URL of the Qdrant service.
        qdrant_api_key: Optional API key used to authenticate with Qdrant.
        search_api_url: Base URL of the search API service.
        search_api_key: Optional API key used to authenticate with the search API.
        embedder_url: Base URL of the embedding service.
        embedder_api_key: Optional API key used to authenticate with the embedding service.
        embedding_dimensions: Number of dimensions produced by the embedding model.
        chunk_max_chars: Maximum number of characters allowed in each text chunk.
        chunk_overlap: Number of trailing characters carried into the next text chunk.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        env_parse_none_str="null",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
        str_min_length=1,
        str_strip_whitespace=True,
        validate_default=True,
    )

    qdrant_url: Annotated[
        HttpUrl,
        Field(description="Base URL of the Qdrant service.", validation_alias=AliasChoices("QDRANT_URL")),
    ] = DEFAULT_QDRANT_URL

    qdrant_api_key: SecretStr | None = Field(
        default=None,
        description="Optional API key used to authenticate with Qdrant.",
        validation_alias=AliasChoices("QDRANT_API_KEY"),
    )

    search_api_url: Annotated[
        HttpUrl,
        Field(description="Base URL of the search API service.", validation_alias=AliasChoices("SEARCH_API_URL")),
    ] = DEFAULT_SEARCH_API_URL
    search_api_key: SecretStr | None = Field(
        default=None,
        description="Optional API key used to authenticate with the search API.",
        validation_alias=AliasChoices("SEARCH_API_KEY"),
    )

    embedder_url: Annotated[
        HttpUrl,
        Field(description="Base URL of the embedding service.", validation_alias=AliasChoices("EMBEDDER_URL")),
    ] = DEFAULT_EMBEDDER_URL
    embedder_api_key: SecretStr | None = Field(
        default=None,
        description="Optional API key used to authenticate with the embedding service.",
        validation_alias=AliasChoices("EMBEDDER_API_KEY"),
    )
    embedding_dimensions: EmbeddingDimensions = Field(
        default=384,
        description="Number of dimensions produced by the embedding model.",
        validation_alias=AliasChoices("EMBEDDING_DIMENSIONS"),
    )

    chunk_max_chars: ChunkMaxChars = Field(
        default=DEFAULT_CHUNK_MAX_CHARS,
        description="Maximum number of characters allowed in each text chunk.",
        validation_alias=AliasChoices("CHUNK_MAX_CHARS"),
    )
    chunk_overlap: ChunkOverlap = Field(
        default=DEFAULT_CHUNK_OVERLAP,
        description="Number of trailing characters carried into the next text chunk.",
        validation_alias=AliasChoices("CHUNK_OVERLAP"),
    )
    collection_prefix: str = Field(
        default="context9",
        description="Prefix used when creating Qdrant collection names.",
        validation_alias=AliasChoices("COLLECTION_PREFIX"),
    )
    qdrant_timeout_seconds: int = Field(
        default=10,
        ge=1,
        description="Qdrant client timeout in seconds.",
        validation_alias=AliasChoices("QDRANT_TIMEOUT_SECONDS"),
    )

    @model_validator(mode="after")
    def _validate_chunk_window(self) -> Self:
        if self.chunk_overlap >= self.chunk_max_chars:
            msg = "chunk_overlap must be less than chunk_max_chars."
            raise ValueError(msg)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()
