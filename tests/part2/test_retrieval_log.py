"""Tests for the retrieval log: records user queries and the text chunks retrieved."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.part2.rag.pipeline import RAGPipeline, NO_INFO_MSG
from .conftest import EXAMPLE_DIR, CONFIG_DIR

# Integration test writes here so results persist for inspection
LOG_PATH = Path(__file__).parent / "retrieval_log.json"

SAMPLE_QUERIES = [
    "Which section describes Fig. 4?",
    "What subsections are in the Background?",
    "Which section includes the description of Table 6?",
]


def write_retrieval_log(pipeline: RAGPipeline, queries: list[str], out_path: Path) -> list[dict]:
    """Run each query through the pipeline and write query + chunks to a JSON log."""
    records = []
    for question in queries:
        result = pipeline.query(question)
        records.append({
            "query": question,
            "chunks": result["retrieved_chunks"],
            "best_score": result["best_score"],
        })
    out_path.write_text(json.dumps(records, indent=2))
    return records


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_pipeline(config_path) -> RAGPipeline:
    with patch("src.part2.rag.pipeline.get_client") as mock_get:
        mock_get.return_value = (MagicMock(), "gemma3:4b")
        return RAGPipeline(config_path)


# ── Unit tests ─────────────────────────────────────────────────────────────────

def test_retrieval_log_file_created(default_config_path, tmp_path):
    """Log file is created and contains one entry per query."""
    p = _make_pipeline(default_config_path)
    log_path = tmp_path / "retrieval_log.json"
    write_retrieval_log(p, SAMPLE_QUERIES, log_path)
    assert log_path.exists()
    data = json.loads(log_path.read_text())
    assert len(data) == len(SAMPLE_QUERIES)


def test_retrieval_log_entry_structure(default_config_path, tmp_path):
    """Each log entry has query, chunks, and best_score fields."""
    p = _make_pipeline(default_config_path)
    log_path = tmp_path / "retrieval_log.json"
    records = write_retrieval_log(p, ["What is Fig. 4?"], log_path)
    entry = records[0]
    assert "query" in entry
    assert "chunks" in entry
    assert "best_score" in entry
    assert isinstance(entry["chunks"], list)


def test_retrieval_log_chunk_fields(default_config_path, tmp_path):
    """Chunks in the log contain score, text, and pdf_id."""
    p = _make_pipeline(default_config_path)
    fake_resp = MagicMock()
    fake_resp.choices[0].message.content = "answer"
    p.client.chat.completions.create.return_value = fake_resp
    with patch.object(
        p.text_index, "search",
        return_value=[(0.9, {"text": "section content about the figure", "pdf_id": "23870758"})]
    ):
        log_path = tmp_path / "retrieval_log.json"
        records = write_retrieval_log(p, ["Which section describes Fig. 4?"], log_path)
    chunk = records[0]["chunks"][0]
    assert "score" in chunk
    assert "text" in chunk
    assert "pdf_id" in chunk


def test_retrieval_log_below_threshold_has_empty_chunks(default_config_path, tmp_path):
    """Queries below the similarity threshold produce an entry with an empty chunks list."""
    p = _make_pipeline(default_config_path)
    log_path = tmp_path / "retrieval_log.json"
    records = write_retrieval_log(p, ["xyzzy nonsense query"], log_path)
    assert records[0]["chunks"] == []
    assert records[0]["best_score"] < p._threshold


# ── Integration test ───────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.skipif(not EXAMPLE_DIR.exists(), reason="Example data not found")
def test_retrieval_log_integration(default_config_path, tmp_path, ollama_server):
    """Index real PDFs, run sample queries, and write retrieval_log.json for inspection."""
    p = RAGPipeline(default_config_path)
    p._parse_dir = tmp_path / "parsed"
    p._index_dir = tmp_path / "index"
    try:
        p.index_documents(EXAMPLE_DIR)
    except ValueError as e:
        if "architecture" in str(e):
            pytest.skip(f"Docling layout model incompatible: {e}")
        raise

    records = write_retrieval_log(p, SAMPLE_QUERIES, LOG_PATH)

    for record in records:
        print(f"\nQ: {record['query']}")
        print(f"   best_score: {record['best_score']:.3f}")
        for c in record["chunks"]:
            print(f"   [{c['score']:.3f}] ({c['pdf_id']}) {c['text'][:120]} …")

    assert LOG_PATH.exists()
    for record in records:
        assert "query" in record
        assert "chunks" in record
