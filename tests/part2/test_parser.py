"""Tests for the Docling PDF parser."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import src.part2.rag.parser as parser_module
from src.part2.rag.parser import ParsedDoc

EXAMPLE_PDF = Path(__file__).parents[2] / "src" / "part2" / "example_data" / "23870758.pdf"


# ── Unit tests (mocked Docling) ────────────────────────────────────────────────

def _make_mock_result(markdown: str = "# Title\nContent", tables=None, pages=None):
    mock_doc = MagicMock()
    mock_doc.export_to_markdown.return_value = markdown
    mock_doc.tables = tables or []
    mock_doc.pages = pages or {}
    mock_result = MagicMock()
    mock_result.document = mock_doc
    mock_converter = MagicMock()
    mock_converter.convert.return_value = mock_result
    return mock_converter


def test_parse_returns_parseddoc(tmp_path):
    converter = _make_mock_result()
    with patch("src.part2.rag.parser.DocumentConverter", return_value=converter):
        doc = parser_module.parse_pdf(Path("test.pdf"), tmp_path)
    assert isinstance(doc, ParsedDoc)
    assert doc.pdf_id == "test"


def test_parse_markdown_content(tmp_path):
    converter = _make_mock_result(markdown="# Header\nSome text")
    with patch("src.part2.rag.parser.DocumentConverter", return_value=converter):
        doc = parser_module.parse_pdf(Path("test.pdf"), tmp_path)
    assert "Header" in doc.markdown


def test_parse_extracts_tables(tmp_path):
    mock_table = MagicMock()
    mock_table.export_to_markdown.return_value = "| Col1 | Col2 |\n|---|---|\n| A | B |"
    converter = _make_mock_result(tables=[mock_table])
    with patch("src.part2.rag.parser.DocumentConverter", return_value=converter):
        doc = parser_module.parse_pdf(Path("test.pdf"), tmp_path)
    assert len(doc.tables_md) == 1
    assert "Col1" in doc.tables_md[0]


def test_parse_saves_page_images(tmp_path):
    mock_image = MagicMock()
    mock_page = MagicMock()
    mock_page.image = mock_image
    converter = _make_mock_result(pages={1: mock_page})
    with patch("src.part2.rag.parser.DocumentConverter", return_value=converter):
        doc = parser_module.parse_pdf(Path("test.pdf"), tmp_path)
    # PIL save should have been called once
    mock_image.pil_image.save.assert_called_once()


def test_output_dir_created(tmp_path):
    out = tmp_path / "subdir" / "parsed"
    converter = _make_mock_result()
    with patch("src.part2.rag.parser.DocumentConverter", return_value=converter):
        parser_module.parse_pdf(Path("test.pdf"), out)
    assert out.exists()


# ── Integration test (real Docling on real PDF) ────────────────────────────────

@pytest.mark.integration
@pytest.mark.skipif(not EXAMPLE_PDF.exists(), reason="Example PDF not found")
def test_parse_real_pdf(tmp_path):
    try:
        doc = parser_module.parse_pdf(EXAMPLE_PDF, tmp_path / "parsed")
    except ValueError as e:
        if "architecture" in str(e):
            pytest.skip(f"Docling layout model incompatible with installed transformers: {e}")
        raise
    assert isinstance(doc, ParsedDoc)
    assert doc.pdf_id == "23870758"
    assert len(doc.markdown) > 100
