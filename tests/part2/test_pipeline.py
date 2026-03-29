"""Tests for the RAG pipeline: retrieval, threshold gating, and LLM call."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.part2.rag.pipeline import RAGPipeline, NO_INFO_MSG
from .conftest import EXAMPLE_DIR


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_pipeline(config_path) -> RAGPipeline:
    """Return a pipeline with the LLM client mocked out."""
    with patch("src.part2.rag.pipeline.get_client") as mock_get:
        mock_get.return_value = (MagicMock(), "gemma3:4b")
        return RAGPipeline(config_path)


def _llm_reply(pipeline: RAGPipeline, text: str) -> None:
    resp = MagicMock()
    resp.choices[0].message.content = text
    pipeline.client.chat.completions.create.return_value = resp


# ── Unit tests ─────────────────────────────────────────────────────────────────

def test_pipeline_init(default_config_path):
    p = _make_pipeline(default_config_path)
    assert p._top_k > 0
    assert 0 < p._threshold < 1


def test_query_empty_index_returns_no_info(default_config_path):
    """With no documents indexed the similarity is 0 → NO_INFO_MSG."""
    p = _make_pipeline(default_config_path)
    result = p.query("What is the yield percentage of compound 12?")
    assert result["answer"] == NO_INFO_MSG
    assert result["retrieved_chunks"] == []


def test_query_below_threshold(default_config_path):
    p = _make_pipeline(default_config_path)
    with patch.object(
        p.text_index, "search",
        return_value=[(0.01, {"text": "irrelevant", "pdf_id": "doc1"})]
    ):
        result = p.query("random question")
    assert result["answer"] == NO_INFO_MSG
    assert result["best_score"] < p._threshold


def test_query_above_threshold_calls_llm(default_config_path):
    p = _make_pipeline(default_config_path)
    _llm_reply(p, "The yield is 92%.")
    with patch.object(
        p.text_index, "search",
        return_value=[(0.95, {"text": "yield 92%", "pdf_id": "23870758"})]
    ):
        result = p.query("What is the yield?")
    assert result["answer"] == "The yield is 92%."
    assert len(result["retrieved_chunks"]) == 1
    p.client.chat.completions.create.assert_called_once()


def test_query_result_has_required_keys(default_config_path):
    p = _make_pipeline(default_config_path)
    result = p.query("anything")
    for key in ("answer", "retrieved_chunks", "visual_results", "best_score"):
        assert key in result


def test_retrieved_chunks_contain_metadata(default_config_path):
    p = _make_pipeline(default_config_path)
    _llm_reply(p, "answer")
    chunk = {"text": "some text about yield", "pdf_id": "23870758"}
    with patch.object(p.text_index, "search", return_value=[(0.9, chunk)]):
        result = p.query("yield?")
    if result["retrieved_chunks"]:
        assert "pdf_id" in result["retrieved_chunks"][0]
        assert "score" in result["retrieved_chunks"][0]


def test_visual_hits_attached_to_prompt(default_config_path):
    """Images from visual hits should be added to the LLM message content."""
    p = _make_pipeline(default_config_path)
    _llm_reply(p, "answer")
    fake_b64 = "iVBORw0KGgo="  # minimal valid-ish base64
    with (
        patch.object(p.text_index, "search", return_value=[(0.9, {"text": "ctx", "pdf_id": "x"})]),
        patch.object(p.visual_index, "search", return_value=[{"score": 0.85, "base64": fake_b64}]),
    ):
        p.query("test query")

    call_args = p.client.chat.completions.create.call_args
    content = call_args.kwargs["messages"][0]["content"]
    image_parts = [c for c in content if c.get("type") == "image_url"]
    assert len(image_parts) >= 1


# ── Integration test ───────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.skipif(not EXAMPLE_DIR.exists(), reason="Example data not found")
def test_index_and_query_live(default_config_path, tmp_path, ollama_server):
    """Full index + query cycle against a live Ollama server."""
    p = RAGPipeline(default_config_path)
    p._parse_dir = tmp_path / "parsed"
    p._index_dir = tmp_path / "index"
    p.index_documents(EXAMPLE_DIR)
    result = p.query("What tables are present in the document?")
    assert result["answer"]
    assert result["answer"] != NO_INFO_MSG or result["best_score"] < p._threshold
