import math

import pytest

from context9.embeddings import HashEmbeddingService, lexical_score, rerank_candidates
from context9.models import SearchCandidate


def search_candidate_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "chunk-1",
        "text": "FastAPI default response class uses HTMLResponse.",
        "score": 0.25,
        "package": "fastapi",
        "version": "latest",
        "source_url": "https://fastapi.tiangolo.com/advanced/custom-response/",
        "ordinal": 0,
    }
    payload.update(overrides)
    return payload


def test_hash_embedding_service_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions must be greater than zero"):
        HashEmbeddingService(dimensions=0)


def test_hash_embedding_service_rejects_empty_input() -> None:
    service = HashEmbeddingService(dimensions=8)

    with pytest.raises(ValueError, match="at least one text is required"):
        service.embed_texts([])


def test_hash_embedding_service_returns_deterministic_normalized_vectors() -> None:
    service = HashEmbeddingService(dimensions=16)

    first = service.embed_texts(["FastAPI response class"])[0]
    second = service.embed_texts(["FastAPI response class"])[0]

    assert first == second
    assert len(first) == 16
    assert math.sqrt(sum(value * value for value in first)) == pytest.approx(1.0)


def test_hash_embedding_service_returns_zero_vector_for_blank_text() -> None:
    service = HashEmbeddingService(dimensions=4)

    assert service.embed_texts(["   "]) == [[0.0, 0.0, 0.0, 0.0]]


def test_lexical_score_is_case_insensitive_query_term_coverage() -> None:
    score = lexical_score("Default Response Missing", "FastAPI default response class")

    assert score == pytest.approx(2 / 3)


def test_lexical_score_returns_zero_for_empty_query_or_text() -> None:
    assert lexical_score("", "FastAPI") == 0.0
    assert lexical_score("FastAPI", "") == 0.0


def test_rerank_candidates_blends_vector_and_lexical_scores_without_mutating_input() -> None:
    lexical_match = SearchCandidate.model_validate(search_candidate_payload(id="match", score=0.2))
    vector_match = SearchCandidate.model_validate(
        search_candidate_payload(id="vector", text="Unrelated but high vector score.", score=0.5, ordinal=1)
    )

    ranked = rerank_candidates("default response class", [vector_match, lexical_match])

    assert [candidate.id for candidate in ranked] == ["match", "vector"]
    assert ranked[0].score == pytest.approx(0.424)
    assert lexical_match.score == 0.2
