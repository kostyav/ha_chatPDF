"""Orchestrator service — coordinates the PDF indexing pipeline over Redis queues.

Indexing flow:
  1. Push parse.requests for every PDF in pdf_dir.
  2. Wait until both "text" and "visual" index.ready messages arrive for every PDF.
  3. Exit once all PDFs are indexed.

For querying, use query.py (a separate standalone script).
"""
import logging
import os
import sys
import threading
from pathlib import Path

import redis
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

sys.path.insert(0, "/app")
import shared.schemas as schemas

logging.basicConfig(level=logging.INFO, format="%(asctime)s [orchestrator] %(message)s")
log = logging.getLogger(__name__)

REDIS_URL         = os.environ.get("REDIS_URL",    "redis://redis:6379/0")
QDRANT_HOST       = os.environ.get("QDRANT_HOST",  "qdrant")
QDRANT_PORT       = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "rag_chunks"


class Orchestrator:
    def __init__(self) -> None:
        self._r    = redis.from_url(REDIS_URL, decode_responses=True)
        self._r_bg = redis.from_url(REDIS_URL, decode_responses=True)  # dedicated to background thread
        self._r.delete(
            schemas.Q_PARSE_REQUESTS, schemas.Q_INDEX_READY,
        )

        # pdf_id sets confirmed ready per service
        self._index_ready: dict[str, set[str]] = {"text": set(), "visual": set()}
        self._index_event = threading.Event()

        threading.Thread(target=self._consume_index_ready, daemon=True).start()

    # ── Background consumer ────────────────────────────────────────────────────

    def _consume_index_ready(self) -> None:
        log.info("index_ready consumer started")
        while True:
            data = schemas.pop(self._r_bg, schemas.Q_INDEX_READY)
            if data is None:
                continue
            self._index_ready[data["service"]].add(data["pdf_id"])
            log.info("index.ready: %s / %s", data["service"], data["pdf_id"])
            self._index_event.set()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _already_indexed(self, pdf_ids: set[str]) -> set[str]:
        try:
            qdrant   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            existing = {c.name for c in qdrant.get_collections().collections}
            if QDRANT_COLLECTION not in existing:
                return set()
            found = set()
            for pdf_id in pdf_ids:
                results, _ = qdrant.scroll(
                    collection_name=QDRANT_COLLECTION,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="pdf_id", match=MatchValue(value=pdf_id))
                    ]),
                    limit=1,
                )
                if results:
                    found.add(pdf_id)
            return found
        except Exception:
            log.warning("Could not query Qdrant — assuming nothing is indexed", exc_info=True)
            return set()

    # ── Public API ─────────────────────────────────────────────────────────────

    def index_documents(self, pdf_dir: Path) -> None:
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs:
            raise ValueError(f"No PDFs found in {pdf_dir.resolve()}")

        expected = {p.stem for p in pdfs}
        cached   = self._already_indexed(expected)
        to_index = expected - cached

        for pdf_id in cached:
            self._index_ready["text"].add(pdf_id)
            self._index_ready["visual"].add(pdf_id)
        if cached:
            log.info("Skipping %d already-indexed PDF(s): %s", len(cached), sorted(cached))

        if not to_index:
            log.info("All PDFs already indexed — skipping parse pipeline")
            return

        log.info("Sending parse.requests for %d new PDF(s) …", len(to_index))
        for pdf in pdfs:
            if pdf.stem in to_index:
                # parse.requests is consumed by BOTH parser and visual_indexer,
                # so push two copies — one per consumer.
                msg = schemas.parse_request(str(pdf.resolve()), pdf.stem)
                schemas.push(self._r, schemas.Q_PARSE_REQUESTS, msg)
                schemas.push(self._r, schemas.Q_PARSE_REQUESTS, msg)

        while True:
            self._index_event.wait(timeout=5)
            self._index_event.clear()
            text_done   = expected.issubset(self._index_ready["text"])
            visual_done = expected.issubset(self._index_ready["visual"])
            log.info(
                "Index progress — text: %d/%d  visual: %d/%d",
                len(self._index_ready["text"]   & expected), len(expected),
                len(self._index_ready["visual"] & expected), len(expected),
            )
            if text_done and visual_done:
                break


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="RAG orchestrator — index PDFs")
    parser.add_argument(
        "--pdf-dir",
        default=os.environ.get("PDF_DIR"),
        required=not os.environ.get("PDF_DIR"),
        help="Directory of PDFs to index (or set PDF_DIR env var)",
    )
    args = parser.parse_args()

    orch = Orchestrator()
    print(f"Indexing PDFs in {args.pdf_dir} …")
    orch.index_documents(Path(args.pdf_dir))
    print("Index ready.")


if __name__ == "__main__":
    _cli()
