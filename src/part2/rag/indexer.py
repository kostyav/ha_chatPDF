"""Dual-index: FAISS text index (sentence-transformers) + Byaldi visual index (ColQwen2)."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ── Text index (FAISS + sentence-transformers) ─────────────────────────────────

class TextIndex:
    """In-memory FAISS index over text chunks with optional disk persistence."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks: list[dict] = []   # [{text, pdf_id, page_num, ...}]
        self.index: Optional[faiss.Index] = None

    def add(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        embs = self.model.encode(
            [c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        if self.index is None:
            self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 3) -> list[tuple[float, dict]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        q = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, ids = self.index.search(q, k)
        return [(float(scores[0][i]), self.chunks[ids[0][i]]) for i in range(k) if ids[0][i] >= 0]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss.index"))
        (path / "chunks.pkl").write_bytes(pickle.dumps(self.chunks))

    def load(self, path: Path) -> None:
        self.index = faiss.read_index(str(path / "faiss.index"))
        self.chunks = pickle.loads((path / "chunks.pkl").read_bytes())


# ── Visual index (Byaldi + ColQwen2) ──────────────────────────────────────────

class VisualIndex:
    """Byaldi RAGMultiModalModel wrapper for page-level visual retrieval."""

    def __init__(self, model_name: str = "vidore/colqwen2-v0.1"):
        self.model_name = model_name
        self._model = None  # lazy-loaded

    def _load_model(self):
        from byaldi import RAGMultiModalModel  # optional heavy dep
        if self._model is None:
            self._model = RAGMultiModalModel.from_pretrained(
                self.model_name, verbose=0
            )
        return self._model

    def index(self, pdf_dir: Path, index_dir: Path) -> None:
        """Build a Byaldi index over all PDFs in pdf_dir."""
        index_dir.mkdir(parents=True, exist_ok=True)
        model = self._load_model()
        model.index(
            input_path=str(pdf_dir),
            index_name="visual_index",
            index_root=str(index_dir),
            overwrite=True,
        )

    def load(self, index_dir: Path) -> None:
        from byaldi import RAGMultiModalModel
        self._model = RAGMultiModalModel.from_index(
            str(index_dir / "visual_index"), verbose=0
        )

    def search(self, query: str, k: int = 3) -> list[dict]:
        if self._model is None:
            return []
        results = self._model.search(query, k=k, return_base64_results=True)
        return [
            {
                "score": float(r.score),
                "doc_id": getattr(r, "doc_id", None),
                "page_num": getattr(r, "page_num", None),
                "base64": getattr(r, "base64", None),
                "metadata": getattr(r, "metadata", {}),
            }
            for r in results
        ]
