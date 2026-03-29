"""Evaluation script: iterate CSV, run RAG queries, score with BERTScore, log results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from bert_score import score as bert_score

from .rag.pipeline import RAGPipeline


def run(
    csv_path: Path,
    pdf_dir: Path,
    config_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> tuple[list[dict], float]:
    pipeline = RAGPipeline(config_path)
    pipeline.index_documents(pdf_dir)

    df = pd.read_csv(csv_path)
    records: list[dict] = []

    for _, row in df.iterrows():
        result = pipeline.query(str(row["question"]))
        record = {
            "question": row["question"],
            "ground_truth": str(row["answer"]),
            "predicted": result["answer"],
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "best_score": result.get("best_score", 0.0),
            "visual_results": result.get("visual_results", []),
        }
        records.append(record)
        print(f"Q : {row['question'][:80]}")
        print(f"GT: {str(row['answer'])[:120]}")
        print(f"A : {result['answer'][:200]}")
        print(f"chunks: {len(record['retrieved_chunks'])}  best_score: {record['best_score']:.3f}")
        print()

    # BERTScore evaluation
    refs = [r["ground_truth"] for r in records]
    hyps = [r["predicted"] for r in records]
    _, _, F1 = bert_score(hyps, refs, lang="en", verbose=False)

    for i, r in enumerate(records):
        r["bert_f1"] = float(F1[i])

    avg_f1 = float(F1.mean())
    print(f"Average BERTScore F1: {avg_f1:.4f}")

    if out_path:
        out_path.write_text(json.dumps({"avg_bert_f1": avg_f1, "results": records}, indent=2))

    return records, avg_f1


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline against ground-truth CSV")
    parser.add_argument("--csv",     required=True, help="Path to ground-truth CSV")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing source PDFs")
    parser.add_argument("--config",  default=None,  help="Config YAML path (default: config.yaml)")
    parser.add_argument("--output",  default="eval_results.json", help="JSON output path")
    args = parser.parse_args()
    run(Path(args.csv), Path(args.pdf_dir), args.config and Path(args.config), Path(args.output))


if __name__ == "__main__":
    _cli()
