from __future__ import annotations

import pytest

from context9.text import chunk_text, extract_html_text, normalize_text


def test_normalize_text_compacts_horizontal_whitespace() -> None:
    text = "  FastAPI\t\tDepends\r\f\v wiring  "

    assert normalize_text(text) == "FastAPI Depends wiring"


def test_normalize_text_strips_each_line_and_limits_blank_lines() -> None:
    text = "  First line  \n\n\n\n  Second line  \n   \n\n  Third line  "

    assert normalize_text(text) == "First line\n\nSecond line\n\nThird line"


def test_extract_html_text_returns_title_and_main_content() -> None:
    html = """
    <html>
      <head><title> FastAPI Reference </title></head>
      <body>
        <nav>Global navigation</nav>
        <main>
          <h1>FastAPI</h1>
          <p>Create API routes with Depends.</p>
        </main>
        <footer>Legal footer</footer>
      </body>
    </html>
    """

    title, text = extract_html_text(html)

    assert title == "FastAPI Reference"
    assert text == "FastAPI\nCreate API routes with Depends."


def test_extract_html_text_prefers_article_when_main_is_missing() -> None:
    html = """
    <html>
      <body>
        <header>Site header</header>
        <article>
          <h1>Dependencies</h1>
          <p>Declare reusable requirements.</p>
        </article>
        <aside>Related links</aside>
      </body>
    </html>
    """

    title, text = extract_html_text(html)

    assert title is None
    assert text == "Dependencies\nDeclare reusable requirements."


def test_extract_html_text_falls_back_to_body_content() -> None:
    html = """
    <html>
      <body>
        <h1>Body heading</h1>
        <p>Body paragraph.</p>
      </body>
    </html>
    """

    title, text = extract_html_text(html)

    assert title is None
    assert text == "Body heading\nBody paragraph."


def test_extract_html_text_removes_non_content_elements() -> None:
    html = """
    <html>
      <head>
        <title>Docs</title>
        <style>.hidden { display: none; }</style>
        <script>alert("nope")</script>
      </head>
      <body>
        <main>
          <button>Copy</button>
          <svg><text>Icon label</text></svg>
          <noscript>Enable JavaScript</noscript>
          <form>Search form</form>
          <div role="navigation">Breadcrumbs</div>
          <div aria-label="Table of contents">On this page</div>
          <p>Only useful documentation remains.</p>
        </main>
      </body>
    </html>
    """

    title, text = extract_html_text(html)

    assert title == "Docs"
    assert text == "Only useful documentation remains."


def test_extract_html_text_handles_fragment_without_html_document() -> None:
    title, text = extract_html_text("<section><h2>Fragment</h2><p>Still parseable.</p></section>")

    assert title is None
    assert text == "Fragment\nStill parseable."


def test_extract_html_text_handles_empty_document() -> None:
    title, text = extract_html_text("")

    assert title is None
    assert text == ""


def test_chunk_text_returns_empty_list_for_blank_text() -> None:
    assert chunk_text(" \n\n\t ") == []


def test_chunk_text_groups_paragraphs_until_max_chars() -> None:
    text = "alpha\n\nbeta\n\ngamma"

    assert chunk_text(text, max_chars=13, overlap=0) == ["alpha\n\nbeta", "gamma"]


def test_chunk_text_uses_overlap_when_it_fits_next_chunk() -> None:
    text = "alpha beta\n\ngamma"

    assert chunk_text(text, max_chars=14, overlap=4) == ["alpha beta", "beta\n\ngamma"]


def test_chunk_text_does_not_exceed_max_chars_when_overlap_would_not_fit() -> None:
    text = "alpha beta\n\ngamma delta"

    chunks = chunk_text(text, max_chars=12, overlap=6)

    assert chunks == ["alpha beta", "gamma delta"]
    assert all(len(chunk) <= 12 for chunk in chunks)


def test_chunk_text_splits_large_blocks_on_word_boundaries_when_possible() -> None:
    chunks = chunk_text("alpha beta gamma delta", max_chars=12, overlap=0)

    assert chunks == ["alpha beta", "gamma delta"]


def test_chunk_text_splits_large_blocks_with_overlap() -> None:
    chunks = chunk_text("abcdefghijklmnopqrst", max_chars=8, overlap=2)

    assert chunks == ["abcdefgh", "ghijklmn", "mnopqrst"]
    assert all(len(chunk) <= 8 for chunk in chunks)


def test_chunk_text_flushes_current_chunk_before_splitting_large_paragraph() -> None:
    chunks = chunk_text("small\n\nabcdefghijkl", max_chars=5, overlap=0)

    assert chunks == ["small", "abcde", "fghij", "kl"]


@pytest.mark.parametrize(
    ("max_chars", "overlap", "match"),
    [
        (10, 10, "max_chars must be greater than overlap"),
        (0, 0, "max_chars must be greater than zero"),
        (10, -1, "overlap must be greater than or equal to zero"),
    ],
)
def test_chunk_text_rejects_invalid_options(max_chars: int, overlap: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        chunk_text("docs", max_chars=max_chars, overlap=overlap)
