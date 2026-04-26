import pytest
from fastapi import HTTPException, status

from context9.security import verify_api_key


def test_verify_api_key_allows_requests_when_no_key_is_configured() -> None:
    verify_api_key(provided=None, expected=None)
    verify_api_key(provided="anything", expected=None)


def test_verify_api_key_allows_matching_key() -> None:
    verify_api_key(provided="secret", expected="secret")


def test_verify_api_key_rejects_missing_key_when_expected() -> None:
    with pytest.raises(HTTPException) as exc_info:
        verify_api_key(provided=None, expected="secret")

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Invalid API key"


def test_verify_api_key_rejects_mismatched_key() -> None:
    with pytest.raises(HTTPException) as exc_info:
        verify_api_key(provided="wrong", expected="secret")

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Invalid API key"
