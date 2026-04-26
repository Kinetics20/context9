from __future__ import annotations

import re
from hashlib import sha256
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid5

from bs4 import BeautifulSoup

from context9.config import get_settings
from context9.models import DocumentChunk

if TYPE_CHECKING:
    from collections.abc import Iterable

    from bs4.element import Tag

_HORIZONTAL_WHITESPACE_RE = re.compile(r"[^\S\n]+")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")
_NOISE_SELECTORS = (
    "script",
    "style",
    "noscript",
    "svg",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "[role='navigation']",
    "[aria-label='Table of contents']",
)
_NOISE_SELECTOR = ",".join(_NOISE_SELECTORS)
_CONTENT_SELECTORS = ("main", "article")


def normalize_text(text: str) -> str:
    """Normalize text for indexing and chunking.

    Horizontal whitespace is collapsed to single spaces, line edges are
    stripped, and runs of three or more newlines are reduced to one paragraph
    break. Existing paragraph boundaries are preserved.

    Args:
        text: Raw text extracted from HTML, Markdown, or a plain text source.

    Returns:
        Normalized text with stable spacing and paragraph breaks.
    """
    compact_spaces = _HORIZONTAL_WHITESPACE_RE.sub(" ", text)
    compact_lines = "\n".join(line.strip() for line in compact_spaces.splitlines())
    return _EXCESS_BLANK_LINES_RE.sub("\n\n", compact_lines).strip()


def extract_html_text(html: str) -> tuple[str | None, str]:
    """Extract readable text from an HTML document.

    Non-content elements such as navigation, scripts, forms, sidebars, and
    footers are removed before extraction. The body root is selected in a
    documentation-friendly order: ``main`` first, then ``article``, then
    ``body``, and finally the full parsed document for fragments.

    Args:
        html: Raw HTML document or fragment.

    Returns:
        A tuple of ``(title, text)`` where ``title`` is ``None`` when no
        non-empty title is present, and ``text`` is normalized readable content.
    """
    soup = BeautifulSoup(html, "lxml")
    for element in soup.select(_NOISE_SELECTOR):
        element.decompose()

    title = _optional_normalized_text(soup.title.stripped_strings) if soup.title else None
    content = _content_root(soup)
    text = "\n".join(content.stripped_strings)
    return title, normalize_text(text)


def chunk_text(text: str, max_chars: int | None = None, overlap: int | None = None) -> list[str]:
    """Split text into bounded, paragraph-aware chunks.

    The function first normalizes the input, then groups paragraphs until the
    next paragraph would exceed ``max_chars``. When a chunk boundary is needed,
    the next chunk starts with the trailing ``overlap`` characters from the
    previous chunk to preserve retrieval context. A single paragraph longer
    than ``max_chars`` is split into fixed-size overlapping windows.

    Args:
        text: Raw or normalized text to split.
        max_chars: Maximum number of characters allowed in each chunk. When
            omitted, ``Settings.chunk_max_chars`` is used.
        overlap: Number of trailing characters carried into the next chunk.
            When omitted, ``Settings.chunk_overlap`` is used.

    Returns:
        Non-empty chunks with normalized whitespace. Returns an empty list for
        blank input.

    Raises:
        ValueError: If ``max_chars`` is not positive, ``overlap`` is negative,
            or ``overlap`` is greater than or equal to ``max_chars``.
    """
    resolved_max_chars, resolved_overlap = _resolve_chunk_options(max_chars=max_chars, overlap=overlap)

    paragraphs = normalize_text(text).split("\n\n")
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if not paragraph:
            continue
        if len(paragraph) > resolved_max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_large_block(paragraph, resolved_max_chars, resolved_overlap))
            continue

        candidate = _append_paragraph(current, paragraph)
        if len(candidate) <= resolved_max_chars:
            current = candidate
            continue

        chunks.append(current)
        current = _append_paragraph(_overlap_tail(current, resolved_overlap), paragraph)
        if len(current) <= resolved_max_chars:
            continue
        current = paragraph

    if current:
        chunks.append(current)

    return chunks


def build_chunks(
    *,
    package: str,
    version: str,
    source_url: str,
    title: str | None,
    text: str,
    max_chars: int | None = None,
    overlap: int | None = None,
) -> list[DocumentChunk]:
    """Build validated document chunks from source text.

    Args:
        package: Package or project name that owns the documentation.
        version: Documentation version.
        source_url: HTTP URL of the source document.
        title: Optional source title applied to every chunk.
        text: Raw or normalized source text to chunk.
        max_chars: Maximum number of characters allowed in each chunk. When
            omitted, ``Settings.chunk_max_chars`` is used.
        overlap: Number of trailing characters carried into the next chunk.
            When omitted, ``Settings.chunk_overlap`` is used.

    Returns:
        ``DocumentChunk`` instances with stable deterministic IDs, ordinal
        positions, and SHA-256 content metadata.

    Raises:
        ValueError: If chunk size options are invalid.
    """
    pieces = chunk_text(text, max_chars=max_chars, overlap=overlap)
    chunks: list[DocumentChunk] = []
    for ordinal, piece in enumerate(pieces):
        chunks.append(
            DocumentChunk.model_validate(
                {
                    "id": _chunk_id(package, version, source_url, ordinal, piece),
                    "package": package,
                    "version": version,
                    "source_url": source_url,
                    "title": title,
                    "text": piece,
                    "ordinal": ordinal,
                    "metadata": {"content_sha256": _content_sha256(piece)},
                }
            )
        )
    return chunks


def _content_root(soup: BeautifulSoup) -> BeautifulSoup | Tag:
    return next((element for selector in _CONTENT_SELECTORS if (element := soup.find(selector))), soup.body or soup)


def _optional_normalized_text(parts: Iterable[object]) -> str | None:
    text = normalize_text(" ".join(str(part) for part in parts))
    return text or None


def _validate_chunk_options(*, max_chars: int, overlap: int) -> None:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to zero.")
    if max_chars <= overlap:
        raise ValueError("max_chars must be greater than overlap.")


def _resolve_chunk_options(*, max_chars: int | None, overlap: int | None) -> tuple[int, int]:
    settings = get_settings()
    resolved_max_chars = settings.chunk_max_chars if max_chars is None else max_chars
    resolved_overlap = settings.chunk_overlap if overlap is None else overlap
    _validate_chunk_options(max_chars=resolved_max_chars, overlap=resolved_overlap)
    return resolved_max_chars, resolved_overlap


def _append_paragraph(current: str, paragraph: str) -> str:
    return paragraph if not current else f"{current}\n\n{paragraph}"


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    return text[-overlap:].strip()


def _split_large_block(block: str, max_chars: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0

    while start < len(block):  # pragma: no branch
        end = _chunk_end(block, start=start, max_chars=max_chars)
        chunks.append(block[start:end].strip())
        if end == len(block):
            break
        start = max(0, end - overlap)
        while start < len(block) and block[start].isspace():
            start += 1

    return [chunk for chunk in chunks if chunk]


def _chunk_end(block: str, *, start: int, max_chars: int) -> int:
    hard_end = min(start + max_chars, len(block))
    if hard_end == len(block):
        return hard_end

    soft_end = block.rfind(" ", start + 1, hard_end + 1)
    if soft_end <= start:
        return hard_end
    return soft_end


def _chunk_id(package: str, version: str, source_url: str, ordinal: int, text: str) -> str:
    digest = _content_sha256(text)
    return str(uuid5(NAMESPACE_URL, f"{package}:{version}:{source_url}:{ordinal}:{digest}"))


def _content_sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()
