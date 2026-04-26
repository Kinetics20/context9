"""Small authentication helpers for internal APIs."""

from fastapi import HTTPException, status


def verify_api_key(provided: str | None, expected: str | None) -> None:
    """Validate an API key if one is configured.

    Args:
        provided: API key from the request.
        expected: Configured expected API key.

    Raises:
        HTTPException: If the key is required and invalid.
    """
    if expected and provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
