"""Tests for the dependency-free Markdown renderer used by the web report card."""

from __future__ import annotations

from kconv.report_markdown import render_markdown


def test_headings_paragraphs_and_emphasis() -> None:
    html = render_markdown("# Title\n\nA **bold** and _italic_ line.\n\n## Section\n\nTail.")
    assert "<h1>Title</h1>" in html
    assert "<h2>Section</h2>" in html
    assert "<strong>bold</strong>" in html
    assert "<em>italic</em>" in html
    assert "<p>Tail.</p>" in html


def test_lists_quotes_rules_and_code() -> None:
    html = render_markdown(
        "- one\n- two\n\n1. first\n2. second\n\n> quoted\n\n---\n\n"
        "```python\nprint('<unsafe>')\n```\n\nInline `a < b`."
    )
    assert "<ul><li>one</li><li>two</li></ul>" in html
    assert "<ol><li>first</li><li>second</li></ol>" in html
    assert "<blockquote><p>quoted</p></blockquote>" in html
    assert "<hr>" in html
    assert "<pre><code>print('&lt;unsafe&gt;')</code></pre>" in html
    assert "<code>a &lt; b</code>" in html


def test_tables_with_alignment_and_inline_markup() -> None:
    html = render_markdown(
        "| Model | OOF | Note |\n"
        "| --- | ---: | :---: |\n"
        "| `lgbm` | 0.958 | **best** |\n"
        "| histgb | 0.957 | stable |\n"
    )
    assert '<div class="md-table-wrap"><table>' in html
    assert '<th class="md-right">OOF</th>' in html
    assert '<th class="md-center">Note</th>' in html
    assert "<td class=\"md-right\">0.958</td>" in html
    assert "<code>lgbm</code>" in html
    assert "<strong>best</strong>" in html
    assert html.count("<tr>") == 3


def test_links_are_scheme_checked_and_escaped() -> None:
    html = render_markdown(
        "[docs](https://example.com/a?x=1&y=2) and [mail](mailto:a@example.com) "
        "and [bad](javascript:alert(1)) and [relative](../secret)."
    )
    assert 'href="https://example.com/a?x=1&amp;y=2"' in html
    assert 'href="mailto:a@example.com"' in html
    assert "<a " not in html.split("and [bad]")[1]
    assert "javascript:alert(1)" in html
    assert "../secret" in html


def test_raw_html_never_passes_through() -> None:
    html = render_markdown(
        "# <script>alert('x')</script>\n\n"
        "<img src=x onerror=alert(1)>\n\n"
        "<div onclick=\"evil()\">text</div>\n"
    )
    assert "<script>" not in html
    assert "<img" not in html
    assert "<div" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html


def test_empty_input_and_unclosed_fence() -> None:
    assert render_markdown(None) == ""
    assert render_markdown("") == ""
    html = render_markdown("```\n<tag>\n")
    assert html == "<pre><code>&lt;tag&gt;\n</code></pre>"


def test_links_keep_markdown_characters_out_of_href() -> None:
    html = render_markdown(
        "[url](https://example.com/**path**) and [**bold**](https://example.com/plain)"
    )
    assert 'href="https://example.com/**path**"' in html
    assert "<strong>bold</strong>" in html
    href = html.split('href="', 1)[1].split('"', 1)[0]
    assert "<strong>" not in href


def test_mismatched_table_columns_do_not_crash() -> None:
    html = render_markdown("| a | b |\n| --- |\n| 1 | 2 |\n")
    assert "<table>" in html
    assert "<th>a</th><th>b</th>" in html
    assert "<td>1</td><td>2</td>" in html


def test_real_report_shape() -> None:
    report = (
        "# kconv report - demo\n\n"
        "- Recipe: `pesnik/baseline`\n\n"
        "## Models\n\n"
        "| Model | OOF score |\n| --- | ---: |\n| lgbm | 0.9583 |\n"
    )
    html = render_markdown(report)
    assert html.startswith("<h1>kconv report - demo</h1>")
    assert "<table>" in html
    assert "<td class=\"md-right\">0.9583</td>" in html
