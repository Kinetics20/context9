from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from context9.fetcher import RawDocument, fetch_document

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_fetch_document_reads_local_markdown_file(tmp_path: Path) -> None:
    source = tmp_path / "guide.md"
    source.write_text("# Guide\n\nUse Depends.", encoding="utf-8")

    document = await fetch_document(str(source))

    assert document == RawDocument(
        source=str(source),
        body="# Guide\n\nUse Depends.",
        content_type="text/markdown",
    )


@pytest.mark.asyncio
async def test_fetch_document_reads_local_file_with_guessed_content_type(tmp_path: Path) -> None:
    source = tmp_path / "feed.xml"
    source.write_text("<feed />", encoding="utf-8")

    document = await fetch_document(str(source))

    assert document.content_type in {"application/xml", "text/xml"}


@pytest.mark.asyncio
async def test_fetch_document_rejects_missing_local_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        await fetch_document(str(tmp_path / "missing.md"))


@pytest.mark.asyncio
async def test_fetch_document_rejects_local_directory(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError, match="not a file"):
        await fetch_document(str(tmp_path))


@pytest.mark.asyncio
async def test_fetch_document_reads_http_url() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://docs.example.com/page"
        return httpx.Response(
            status_code=200,
            text="<h1>Docs</h1>",
            headers={"content-type": "text/html; charset=utf-8"},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        document = await fetch_document("https://docs.example.com/page", client=client)

    assert document == RawDocument(
        source="https://docs.example.com/page",
        body="<h1>Docs</h1>",
        content_type="text/html; charset=utf-8",
    )


@pytest.mark.asyncio
async def test_fetch_document_creates_short_lived_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAsyncClient:
        def __init__(self, *, follow_redirects: bool, timeout: httpx.Timeout) -> None:
            assert follow_redirects is True
            assert timeout == httpx.Timeout(30.0)

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(
            self,
            exc_type: object,
            exc_value: object,
            traceback: object,
        ) -> None:
            return None

        async def get(self, url: str) -> httpx.Response:
            assert url == "https://docs.example.com/page"
            return httpx.Response(status_code=200, text="Docs", request=httpx.Request("GET", url))

    monkeypatch.setattr("context9.fetcher.httpx.AsyncClient", FakeAsyncClient)

    document = await fetch_document("https://docs.example.com/page")

    assert document == RawDocument(
        source="https://docs.example.com/page",
        body="Docs",
        content_type="text/plain; charset=utf-8",
    )


@pytest.mark.asyncio
async def test_fetch_document_raises_for_http_error() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(status_code=404))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await fetch_document("https://docs.example.com/missing", client=client)
