"""Orchestrator service — coordinates indexing and query pipeline over Redis queues.

Indexing flow:
  1. Push parse.requests for every PDF in pdf_dir.
  2. Wait until both "text" and "visual" index.ready messages arrive for every PDF.

Query flow:
  1. Assign a correlation_id; result queues are per-correlation Redis lists.
  2. Push retrieve.text.requests and retrieve.visual.requests.
  3. BRPOP both result lists (blocking, with timeout).
  4. Apply similarity threshold; if above, call the LLM with text + image context.
"""
import logging
import os
import sys
import threading
import uuid
from pathlib import Path

import redis
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

sys.path.insert(0, "/app")
import shared.schemas as schemas

logging.basicConfig(level=logging.INFO, format="%(asctime)s [orchestrator] %(message)s")
log = logging.getLogger(__name__)

REDIS_URL            = os.environ.get("REDIS_URL",            "redis://redis:6379/0")
QDRANT_HOST          = os.environ.get("QDRANT_HOST",          "qdrant")
QDRANT_PORT          = int(os.environ.get("QDRANT_PORT",      "6333"))
QDRANT_COLLECTION    = "rag_chunks"
LLM_BASE_URL         = os.environ.get("LLM_BASE_URL",         "http://ollama:11434/v1")
LLM_MODEL            = os.environ.get("LLM_MODEL",            "gemma3:4b")
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.3"))
TOP_K                = int(os.environ.get("TOP_K",                  "3"))
RETRIEVE_TIMEOUT     = int(os.environ.get("RETRIEVE_TIMEOUT",       "60"))

NO_INFO_MSG = "The document does not contain information about this query."


class Orchestrator:
    def __init__(self) -> None:
        self._r      = redis.from_url(REDIS_URL, decode_responses=True)
        self._r_bg   = redis.from_url(REDIS_URL, decode_responses=True)  # dedicated to background thread
        self._r.delete(
            schemas.Q_PARSE_REQUESTS, schemas.Q_INDEX_READY,
            schemas.Q_RETRIEVE_TEXT_REQ, schemas.Q_RETRIEVE_TEXT_RES,
            schemas.Q_RETRIEVE_VIS_REQ,  schemas.Q_RETRIEVE_VIS_RES,
        )
        self._llm = OpenAI(base_url=LLM_BASE_URL, api_key="none")

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

    def query(self, question: str) -> dict:
        corr_id      = str(uuid.uuid4())
        text_res_q   = f"res.text.{corr_id}"
        visual_res_q = f"res.visual.{corr_id}"
        log.info("Query [%s]: %s", corr_id[:8], question[:80])

        text_req   = schemas.retrieve_request(corr_id, question, TOP_K)
        visual_req = schemas.retrieve_request(corr_id, question, TOP_K)
        text_req["reply_to"]   = text_res_q
        visual_req["reply_to"] = visual_res_q

        schemas.push(self._r, schemas.Q_RETRIEVE_TEXT_REQ, text_req)
        schemas.push(self._r, schemas.Q_RETRIEVE_VIS_REQ,  visual_req)
        log.info("Retrieve requests pushed, waiting for results …")

        # Use two dedicated connections so both BRPOPs run concurrently in threads
        import json
        text_hits:   list = []
        visual_hits: list = []
        errors: list = []

        def _wait_text():
            r = redis.from_url(REDIS_URL, decode_responses=True)
            res = r.brpop(text_res_q, timeout=RETRIEVE_TIMEOUT)
            if res:
                text_hits.extend(json.loads(res[1])["hits"])
                log.info("Text results received (%d hits)", len(text_hits))
            else:
                log.warning("Text retrieval timeout")

        def _wait_visual():
            r = redis.from_url(REDIS_URL, decode_responses=True)
            res = r.brpop(visual_res_q, timeout=RETRIEVE_TIMEOUT)
            if res:
                visual_hits.extend(json.loads(res[1])["hits"])
                log.info("Visual results received (%d hits)", len(visual_hits))
            else:
                log.warning("Visual retrieval timeout")

        t1 = threading.Thread(target=_wait_text)
        t2 = threading.Thread(target=_wait_visual)
        t1.start(); t2.start()
        t1.join(timeout=RETRIEVE_TIMEOUT + 5)
        t2.join(timeout=RETRIEVE_TIMEOUT + 5)

        self._r.delete(text_res_q, visual_res_q)

        best_score = max(
            [h["score"] for h in text_hits]
            + [h["score"] for h in visual_hits]
            + [0.0]
        )

        if best_score < SIMILARITY_THRESHOLD:
            return {
                "answer":           NO_INFO_MSG,
                "retrieved_chunks": [],
                "visual_results":   [],
                "best_score":       best_score,
            }

        text_ctx = "\n\n---\n\n".join(h["text"] for h in text_hits)
        content: list[dict] = [
            {"type": "text", "text": f"Context:\n{text_ctx}\n\nQuestion: {question}"}
        ]
        for hit in visual_hits[:2]:
            b64 = hit.get("base64")
            if b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })

        resp = self._llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
        )
        return {
            "answer": resp.choices[0].message.content.strip(),
            "retrieved_chunks": [
                {"score": h["score"], "text": h["text"][:300], "pdf_id": h["pdf_id"]}
                for h in text_hits
            ],
            "visual_results": [
                {"score": h["score"], "doc_id": h.get("doc_id"), "page_num": h.get("page_num")}
                for h in visual_hits
            ],
            "best_score": best_score,
        }


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="RAG orchestrator — index PDFs and query")
    parser.add_argument(
        "--pdf-dir",
        default=os.environ.get("PDF_DIR"),
        required=not os.environ.get("PDF_DIR"),
        help="Directory of PDFs to index (or set PDF_DIR env var)",
    )
    parser.add_argument("--question", default=None, help="Single question (omit for REPL)")
    args = parser.parse_args()

    orch = Orchestrator()
    print(f"Indexing PDFs in {args.pdf_dir} …")
    orch.index_documents(Path(args.pdf_dir))
    print("Index ready.\n")

    def _ask(q: str) -> None:
        result = orch.query(q)
        print(f"Answer : {result['answer']}")
        print(f"Score  : {result['best_score']:.3f}")
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
