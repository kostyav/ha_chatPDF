"""RAG pipeline: parse → index → retrieve → generate."""
from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Optional

import yaml

from ..engines.factory import get_client, load_config
from .indexer import TextIndex, VisualIndex
from .parser import ParsedDoc, parse_pdf


def _split_markdown(text: str, max_chars: int = 800) -> list[str]:
    """Split markdown into section-level chunks, then by size if a section is too large."""
    # Split on level 1-3 headers, keeping the header with its content
    sections = re.split(r'(?=^#{1,3} )', text, flags=re.MULTILINE)
    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= max_chars:
            chunks.append(section)
        else:
            # Further split oversized sections by paragraph
            paragraphs = re.split(r'\n\n+', section)
            buf = ""
            for para in paragraphs:
                if buf and len(buf) + len(para) + 2 > max_chars:
                    chunks.append(buf.strip())
                    buf = para
                else:
                    buf = (buf + "\n\n" + para).strip() if buf else para
            if buf:
                chunks.append(buf.strip())
    return chunks or [text[:max_chars]]

NO_INFO_MSG = "The document does not contain information about this query."


class RAGPipeline:
    """End-to-end RAG pipeline using Docling + ColQwen2/Byaldi + sentence-transformers + LLM."""

    def __init__(self, config_path: Optional[Path] = None):
        cfg_path = Path(config_path or Path(__file__).parents[1] / "config" / "config.yaml")
        self.cfg = load_config(cfg_path)
        self.client, self.model = get_client(cfg_path)

        emb_model = self.cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.text_index = TextIndex(emb_model)
        self.visual_index = VisualIndex(self.cfg.get("colqwen_model", "vidore/colqwen2-v0.1"))

        ret = self.cfg.get("retriever", {})
        self._top_k: int = ret.get("top_k", 3)
        self._threshold: float = ret.get("similarity_threshold", 0.3)
        self._index_dir = Path(ret.get("index_dir", ".byaldi_index"))

        par = self.cfg.get("parser", {})
        self._dpi: int = par.get("dpi", 300)
        self._parse_dir = Path(par.get("output_dir", ".parsed_docs"))

        self.parsed: dict[str, ParsedDoc] = {}

    # ── Indexing ───────────────────────────────────────────────────────────────

    def index_documents(self, pdf_dir: Path) -> None:
        """Parse all PDFs in pdf_dir and build text + visual indices."""
        pdfs = sorted(pdf_dir.glob("*.pdf"))

        for pdf_path in pdfs:
            doc = parse_pdf(pdf_path, self._parse_dir / pdf_path.stem, self._dpi)
            self.parsed[doc.pdf_id] = doc
            # Split document into section-level chunks for better retrieval precision
            tables_ctx = "\n\n".join(doc.tables_md)
            full_text = doc.markdown + ("\n\n" + tables_ctx if tables_ctx else "")
            section_chunks = _split_markdown(full_text)
            self.text_index.add([
                {"text": chunk, "pdf_id": doc.pdf_id, "page_num": 0}
                for chunk in section_chunks
            ])

        # Visual index (heavy — skipped if Byaldi unavailable)
        try:
            self.visual_index.index(pdf_dir, self._index_dir)
        except ImportError:
            pass  # byaldi not installed; fall back to text-only retrieval

    # ── Querying ───────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """Retrieve relevant context and generate an answer."""
        text_hits = self.text_index.search(question, k=self._top_k)
        visual_hits = self.visual_index.search(question, k=self._top_k)

        best_score = max(
            [s for s, _ in text_hits] + [h["score"] for h in visual_hits] + [0.0]
        )

        if best_score < self._threshold:
            return {
                "answer": NO_INFO_MSG,
                "retrieved_chunks": [],
                "visual_results": [],
                "best_score": best_score,
            }

        # Build multimodal message content
        text_ctx = "\n\n---\n\n".join(c["text"] for _, c in text_hits)
        content: list[dict] = [
            {"type": "text", "text": f"Context:\n{text_ctx}\n\nQuestion: {question}"}
        ]

        # Attach up to 2 retrieved page images (keep VRAM usage low)
        for hit in visual_hits[:2]:
            b64 = hit.get("base64")
            if b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
        )
        answer = resp.choices[0].message.content.strip()

        return {
            "answer": answer,
            "retrieved_chunks": [
                {"score": s, "text": c["text"][:300], "pdf_id": c["pdf_id"]}
                for s, c in text_hits
            ],
            "visual_results": [
                {"score": h["score"], "doc_id": h.get("doc_id"), "page_num": h.get("page_num")}
                for h in visual_hits
            ],
            "best_score": best_score,
        }


def _cli() -> None:
    import argparse, json, sys
    parser = argparse.ArgumentParser(description="Index PDFs and query the RAG pipeline")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDF files to index")
    parser.add_argument("--config",  default=None,  help="Config YAML path (default: config.yaml)")
    parser.add_argument("--question", default=None, help="Single question (omit for interactive mode)")
    args = parser.parse_args()

    pipeline = RAGPipeline(args.config and Path(args.config))
    print(f"Indexing PDFs in {args.pdf_dir} …")
    pipeline.index_documents(Path(args.pdf_dir))
    print("Index ready.\n")

    def _ask(question: str) -> None:
        result = pipeline.query(question)
        print(f"Answer : {result['answer']}")
        print(f"Score  : {result['best_score']:.3f}")
        print("Chunks :")
        for c in result["retrieved_chunks"]:
            print(f"  [{c['score']:.3f}] ({c['pdf_id']}) {c['text'][:120]} …")
        print()

    if args.question:
        _ask(args.question)
    else:
        print("Interactive mode — type a question or 'quit' to exit.")
        for line in sys.stdin:
            q = line.strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                _ask(q)


if __name__ == "__main__":
    _cli()
