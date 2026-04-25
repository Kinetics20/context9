from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Annotated

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

DEFAULT_TIMEOUT = httpx.Timeout(30.0)
DEFAULT_CONTENT_TYPE = "text/plain"

type NonEmptyStr = Annotated[str, Field(min_length=1)]


class RawDocument(BaseModel):
    """Fetched source document.

    Attributes:
        source: Original URL or local path used to fetch the document.
        body: Decoded document body.
        content_type: Best-known content type for the document.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_min_length=1,
        str_strip_whitespace=True,
        validate_default=True,
    )

    source: NonEmptyStr
    body: NonEmptyStr
    content_type: str | None = None


async def fetch_document(source: NonEmptyStr, *, client: httpx.AsyncClient | None = None) -> RawDocument:
    """Fetch a document from an HTTP(S) URL or local file path.

    Args:
        source: HTTP(S) URL or local filesystem path.
        client: Optional HTTPX async client. When omitted, a short-lived client is created.

    Returns:
        Fetched document body and content metadata.
    """
    if _is_http_source(source):
        return await _fetch_http_document(source, client=client)

    return _fetch_file_document(Path(source))


async def _fetch_http_document(source: NonEmptyStr, *, client: httpx.AsyncClient | None) -> RawDocument:
    url = HttpUrl(source)
    if client is not None:
        return await _fetch_with_client(client, str(url), source=source)

    async with httpx.AsyncClient(follow_redirects=True, timeout=DEFAULT_TIMEOUT) as owned_client:
        return await _fetch_with_client(owned_client, str(url), source=source)


async def _fetch_with_client(client: httpx.AsyncClient, url: str, *, source: str) -> RawDocument:
    response = await client.get(url)
    response.raise_for_status()
    return RawDocument(source=source, body=response.text, content_type=response.headers.get("content-type"))


def _fetch_file_document(path: Path) -> RawDocument:
    if not path.exists():
        msg = f"Document source does not exist: {path}"
        raise FileNotFoundError(msg)
    if not path.is_file():
        msg = f"Document source is not a file: {path}"
        raise IsADirectoryError(msg)

    return RawDocument(
        source=str(path),
        body=path.read_text(encoding="utf-8"),
        content_type=_content_type_for_path(path),
    )


def _is_http_source(source: str) -> bool:
    return source.startswith(("http://", "https://"))


CONTENT_TYPES: dict[str, str] = {
    ".html": "text/html",
    ".htm": "text/html",
    ".json": "application/json",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}


def _content_type_for_path(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or DEFAULT_CONTENT_TYPE
