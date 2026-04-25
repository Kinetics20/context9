import pytest
from pydantic import ValidationError

from context9.models import (
    DocumentChunk,
    EmbedRequest,
    EmbedResponse,
    IngestReport,
    RerankResponse,
    SearchCandidate,
    SearchResponse,
)


def document_chunk_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "chunk-1",
        "package": "fastapi",
        "source_url": "https://fastapi.tiangolo.com/tutorial/dependencies/",
        "text": "FastAPI dependency injection example.",
        "ordinal": 0,
    }
    payload.update(overrides)
    return payload


def search_candidate_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "chunk-1",
        "text": "Dependency injection",
        "score": 0.98,
        "package": "fastapi",
        "version": "latest",
        "source_url": "https://fastapi.tiangolo.com/tutorial/dependencies/",
        "ordinal": 0,
    }
    payload.update(overrides)
    return payload


def test_document_chunk_strips_strings_and_validates_url() -> None:
    chunk = DocumentChunk.model_validate(
        document_chunk_payload(
            id="  chunk-1  ",
            package="  fastapi  ",
            title="  Dependencies  ",
            text="  Depends wires providers into endpoints.  ",
        )
    )

    assert chunk.id == "chunk-1"
    assert chunk.package == "fastapi"
    assert chunk.title == "Dependencies"
    assert chunk.text == "Depends wires providers into endpoints."
    assert str(chunk.source_url) == "https://fastapi.tiangolo.com/tutorial/dependencies/"


@pytest.mark.parametrize(
    "overrides",
    [
        {"extra_field": "nope"},
        {"text": "   "},
        {"source_url": "not a url"},
        {"ordinal": "0"},
        {"ordinal": -1},
    ],
)
def test_document_chunk_rejects_invalid_payloads(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        DocumentChunk.model_validate(document_chunk_payload(**overrides))


def test_document_chunk_metadata_accepts_nested_values_and_serializes_to_json() -> None:
    chunk = DocumentChunk.model_validate(
        document_chunk_payload(metadata={"section": "DI", "tags": ["fastapi", {"kind": "docs"}]})
    )

    assert chunk.model_dump(mode="json")["metadata"] == {
        "section": "DI",
        "tags": ["fastapi", {"kind": "docs"}],
    }


def test_embed_request_rejects_empty_text_list() -> None:
    with pytest.raises(ValidationError):
        EmbedRequest.model_validate({"texts": []})


def test_embed_response_validates_vector_dimensions() -> None:
    response = EmbedResponse.model_validate({"dimensions": 2, "vectors": [[0.1, 0.2], [0.3, 0.4]]})

    assert response.model_dump() == {"dimensions": 2, "vectors": [[0.1, 0.2], [0.3, 0.4]]}


def test_embed_response_rejects_mismatched_vector_dimensions() -> None:
    with pytest.raises(ValidationError, match="declared dimensions"):
        EmbedResponse.model_validate({"dimensions": 2, "vectors": [[0.1]]})


def test_search_candidate_uses_ordinal_and_rejects_legacy_typo() -> None:
    candidate = SearchCandidate.model_validate(search_candidate_payload())

    assert candidate.ordinal == 0
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SearchCandidate.model_validate(search_candidate_payload(orginal=0, ordinal=None))


def test_rerank_response_uses_results_and_rejects_legacy_typo() -> None:
    candidate = SearchCandidate.model_validate(search_candidate_payload())

    response = RerankResponse.model_validate({"results": [candidate.model_dump()]})

    assert response.results == [candidate]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RerankResponse.model_validate({"resutls": [candidate.model_dump()]})


def test_search_response_requires_non_empty_results() -> None:
    with pytest.raises(ValidationError):
        SearchResponse.model_validate({"package": "fastapi", "version": "latest", "query": "Depends", "results": []})


def test_ingest_report_rejects_negative_chunk_count() -> None:
    with pytest.raises(ValidationError):
        IngestReport.model_validate(
            {
                "package": "fastapi",
                "version": "latest",
                "source": "https://fastapi.tiangolo.com/",
                "chunks": -1,
                "collection": "docs",
            }
        )
