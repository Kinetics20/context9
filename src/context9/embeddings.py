"""Embedding and re-ranking services."""

from __future__ import annotations

import math
import re
from hashlib import blake2b
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from context9.models import SearchCandidate

_TOKEN_RE = re.compile(r"\w+")
_HASH_PERSON = b"context9"
_LEXICAL_WEIGHT = 0.28
_VECTOR_WEIGHT = 0.72


class EmbeddingService(Protocol):
    """Interface implemented by embedding providers."""

    dimensions: int

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a list of texts.

        Args:
            texts: Non-empty texts.

        Returns:
            Dense vectors, one per text.
        """


class HashEmbeddingService:
    """Fast deterministic embedder for local development and tests.

    This is not a replacement for a neural model. It gives the system a
    production-shaped boundary while staying lightweight on Python 3.14.
    """

    def __init__(self, dimensions: int = 384) -> None:
        """Initialize the hash embedder.

        Args:
            dimensions: Number of vector dimensions.

        Raises:
            ValueError: If dimensions is not positive.
        """
        if dimensions <= 0:
            msg = "dimensions must be greater than zero"
            raise ValueError(msg)
        self.dimensions = dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text using signed feature hashing.

        Args:
            texts: Texts to embed.

        Returns:
            L2-normalized dense vectors.

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            msg = "at least one text is required"
            raise ValueError(msg)
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        """Embed one text."""
        vector = [0.0] * self.dimensions
        tokens = _tokens(text)
        for feature in (*tokens, *_bigrams(tokens)):
            digest = blake2b(feature.encode("utf-8"), digest_size=16, person=_HASH_PERSON).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            weight = 1.0 + digest[5] / 255.0
            vector[index] += sign * weight

        return _normalize(vector)


def rerank_candidates(query: str, candidates: Sequence[SearchCandidate]) -> list[SearchCandidate]:
    """Re-rank candidates with lexical overlap blended into vector score.

    Args:
        query: User query.
        candidates: Candidate chunks from the vector database.

    Returns:
        Candidates sorted by blended relevance score.
    """
    ranked: list[SearchCandidate] = []
    for candidate in candidates:
        blended_score = (_VECTOR_WEIGHT * candidate.score) + (_LEXICAL_WEIGHT * lexical_score(query, candidate.text))
        ranked.append(candidate.model_copy(update={"score": round(blended_score, 6)}))
    return sorted(ranked, key=lambda candidate: candidate.score, reverse=True)


def lexical_score(query: str, text: str) -> float:
    """Compute simple token overlap score.

    Args:
        query: User query.
        text: Candidate text.

    Returns:
        Value between 0 and 1.
    """
    query_terms = set(_tokens(query))
    if not query_terms:
        return 0.0
    text_terms = set(_tokens(text))
    if not text_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _bigrams(tokens: Sequence[str]) -> list[str]:
    return [f"{left}_{right}" for left, right in zip(tokens, tokens[1:], strict=False)]


def _normalize(vector: list[float]) -> list[float]:
    """L2-normalize a vector."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
