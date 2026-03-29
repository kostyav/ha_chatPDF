"""Tests for the evaluation script (evaluate.py)."""
import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def mini_csv(tmp_path) -> Path:
    df = pd.DataFrame([
        {"question": "What section describes Fig. 4?", "answer": "Results", "pmcid": "26781090"},
        {"question": "What is the yield of compound 12?", "answer": "92%",   "pmcid": "23870758"},
    ])
    p = tmp_path / "gt.csv"
    df.to_csv(p, index=False)
    return p


def _mock_pipeline(answers: list[str]):
    pipeline = MagicMock()
    side_effects = [
        {
            "answer": a,
            "retrieved_chunks": [{"text": "chunk", "pdf_id": "doc", "score": 0.8}],
            "visual_results": [],
            "best_score": 0.8,
        }
        for a in answers
    ]
    pipeline.query.side_effect = side_effects
    return pipeline


# ── Unit tests ─────────────────────────────────────────────────────────────────

def test_run_returns_records_and_f1(mini_csv, tmp_path, default_config_path):
    mock_pipeline = _mock_pipeline(["Results section", "92 percent"])
    with patch("src.part2.evaluate.RAGPipeline", return_value=mock_pipeline):
        from src.part2.evaluate import run
        records, avg_f1 = run(mini_csv, tmp_path, default_config_path, tmp_path / "out.json")

    assert len(records) == 2
    assert 0.0 <= avg_f1 <= 1.0


def test_run_writes_json_output(mini_csv, tmp_path, default_config_path):
    mock_pipeline = _mock_pipeline(["answer1", "answer2"])
    out = tmp_path / "results.json"
    with patch("src.part2.evaluate.RAGPipeline", return_value=mock_pipeline):
        from src.part2.evaluate import run
        run(mini_csv, tmp_path, default_config_path, out)

    data = json.loads(out.read_text())
    assert "avg_bert_f1" in data
    assert len(data["results"]) == 2


def test_run_each_record_has_bert_f1(mini_csv, tmp_path, default_config_path):
    mock_pipeline = _mock_pipeline(["Results", "92%"])
    with patch("src.part2.evaluate.RAGPipeline", return_value=mock_pipeline):
        from src.part2.evaluate import run
        records, _ = run(mini_csv, tmp_path, default_config_path)

    for r in records:
        assert "bert_f1" in r
        assert 0.0 <= r["bert_f1"] <= 1.0


def test_run_no_info_answers_get_low_score(mini_csv, tmp_path, default_config_path):
    from src.part2.rag.pipeline import NO_INFO_MSG
    mock_pipeline = _mock_pipeline([NO_INFO_MSG, NO_INFO_MSG])
    with patch("src.part2.evaluate.RAGPipeline", return_value=mock_pipeline):
        from src.part2.evaluate import run
        records, avg_f1 = run(mini_csv, tmp_path, default_config_path)

    # NO_INFO_MSG responses should score lower than correct answers
    assert avg_f1 < 0.9


def test_run_skips_json_write_when_out_path_none(mini_csv, tmp_path, default_config_path):
    mock_pipeline = _mock_pipeline(["a", "b"])
    with patch("src.part2.evaluate.RAGPipeline", return_value=mock_pipeline):
        from src.part2.evaluate import run
        records, _ = run(mini_csv, tmp_path, default_config_path, out_path=None)
    assert records  # still returns results
