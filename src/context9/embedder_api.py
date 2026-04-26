"""FastAPI application exposing embedding and re-ranking endpoints."""

from typing import TYPE_CHECKING, Annotated

import uvicorn
from fastapi import Depends, FastAPI, Header

from context9.config import Settings, get_settings
from context9.embeddings import EmbeddingService, HashEmbeddingService, rerank_candidates
from context9.models import EmbedRequest, EmbedResponse, RerankRequest, RerankResponse
from context9.security import verify_api_key

if TYPE_CHECKING:
    from pydantic import SecretStr


def create_app() -> FastAPI:
    """Create the embedder FastAPI application.

    Returns:
        Configured FastAPI app.
    """
    app = FastAPI(title="Context9 Embedder", version="0.1.0")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Return service health."""
        return {"status": "ok"}

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(
        request: EmbedRequest,
        settings: Annotated[Settings, Depends(get_settings)],
        service: Annotated[EmbeddingService, Depends(get_embedding_service)],
        api_key: Annotated[str | None, Header(alias="api-key")] = None,
    ) -> EmbedResponse:
        """Embed request texts."""
        verify_api_key(api_key, _secret_value(settings.embedder_api_key))
        vectors = service.embed_texts(request.texts)
        return EmbedResponse.model_validate({"dimensions": service.dimensions, "vectors": vectors})

    @app.post("/rerank", response_model=RerankResponse)
    async def rerank(
        request: RerankRequest,
        settings: Annotated[Settings, Depends(get_settings)],
        api_key: Annotated[str | None, Header(alias="api-key")] = None,
    ) -> RerankResponse:
        """Re-rank candidate chunks."""
        verify_api_key(api_key, _secret_value(settings.embedder_api_key))
        return RerankResponse(results=rerank_candidates(request.query, request.candidates))

    return app


def get_embedding_service(settings: Annotated[Settings, Depends(get_settings)]) -> EmbeddingService:
    """Return the configured embedding service."""
    return HashEmbeddingService(dimensions=settings.embedding_dimensions)


def _secret_value(secret: SecretStr | None) -> str | None:
    if secret is None:
        return None
    return secret.get_secret_value()


app = create_app()


def run() -> None:
    """Run the embedder API with Uvicorn."""
    uvicorn.run("context9.embedder_api:app", host="0.0.0.0", port=8500)  # noqa: S104
