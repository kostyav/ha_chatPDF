"""Orchestrator service — coordinates indexing and query pipeline over Kafka.

Indexing flow:
  1. Send parse.requests for every PDF in pdf_dir.
  2. Wait until both "text" and "visual" index.ready messages arrive for every PDF.

Query flow:
  1. Assign a correlation_id and register result queues for both indexers.
  2. Publish retrieve.text.requests and retrieve.visual.requests in parallel.
  3. Block until both result messages arrive (or timeout).
  4. Apply similarity threshold; if above, call the LLM with text + image context.
"""
import json
import logging
import os
import queue
import sys
import threading
import uuid
from pathlib import Path

from kafka import KafkaConsumer, KafkaProducer
from openai import OpenAI

sys.path.insert(0, "/app")
import shared.schemas as schemas

logging.basicConfig(level=logging.INFO, format="%(asctime)s [orchestrator] %(message)s")
log = logging.getLogger(__name__)

KAFKA_BOOTSTRAP      = os.environ.get("KAFKA_BOOTSTRAP",      "kafka:9092")
LLM_BASE_URL         = os.environ.get("LLM_BASE_URL",         "http://ollama:11434/v1")
LLM_MODEL            = os.environ.get("LLM_MODEL",            "gemma3:4b")
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.3"))
TOP_K                = int(os.environ.get("TOP_K",                  "3"))
RETRIEVE_TIMEOUT     = float(os.environ.get("RETRIEVE_TIMEOUT",    "60"))

NO_INFO_MSG = "The document does not contain information about this query."


class Orchestrator:
    def __init__(self) -> None:
        self._producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode(),
        )
        self._llm = OpenAI(base_url=LLM_BASE_URL, api_key="none")

        # correlation_id → {"text": Queue, "visual": Queue}
        self._pending: dict[str, dict[str, queue.Queue]] = {}
        self._pending_lock = threading.Lock()

        # pdf_id sets confirmed ready per service
        self._index_ready: dict[str, set[str]] = {"text": set(), "visual": set()}
        self._index_event = threading.Event()

        for target in (
            self._consume_index_ready,
            self._consume_text_results,
            self._consume_visual_results,
        ):
            threading.Thread(target=target, daemon=True).start()

    # ── Background consumers ───────────────────────────────────────────────────

    def _consume_index_ready(self) -> None:
        consumer = KafkaConsumer(
            schemas.TOPIC_INDEX_READY,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id="orchestrator_index",
            value_deserializer=lambda b: json.loads(b.decode()),
            auto_offset_reset="earliest",
        )
        for msg in consumer:
            data = msg.value
            self._index_ready[data["service"]].add(data["pdf_id"])
            log.info("index.ready: %s / %s", data["service"], data["pdf_id"])
            self._index_event.set()

    def _consume_text_results(self) -> None:
        consumer = KafkaConsumer(
            schemas.TOPIC_RETRIEVE_TEXT_RES,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id="orchestrator_text_res",
            value_deserializer=lambda b: json.loads(b.decode()),
            auto_offset_reset="latest",
        )
        for msg in consumer:
            data = msg.value
            with self._pending_lock:
                q = self._pending.get(data["correlation_id"], {}).get("text")
            if q:
                q.put(data["hits"])

    def _consume_visual_results(self) -> None:
        consumer = KafkaConsumer(
            schemas.TOPIC_RETRIEVE_VIS_RES,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id="orchestrator_visual_res",
            value_deserializer=lambda b: json.loads(b.decode()),
            auto_offset_reset="latest",
        )
        for msg in consumer:
            data = msg.value
            with self._pending_lock:
                q = self._pending.get(data["correlation_id"], {}).get("visual")
            if q:
                q.put(data["hits"])

    # ── Public API ─────────────────────────────────────────────────────────────

    def index_documents(self, pdf_dir: Path) -> None:
        """Send parse.requests for all PDFs; block until both indices confirm every PDF."""
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs:
            raise ValueError(f"No PDFs found in {pdf_dir.resolve()}")

        expected = {p.stem for p in pdfs}
        log.info("Sending parse.requests for %d PDFs …", len(pdfs))
        for pdf in pdfs:
            self._producer.send(
                schemas.TOPIC_PARSE_REQUESTS,
                schemas.parse_request(str(pdf.resolve()), pdf.stem),
            )
        self._producer.flush()

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
        """Retrieve context from both indexers, then generate an answer via LLM."""
        corr_id  = str(uuid.uuid4())
        text_q:   queue.Queue = queue.Queue()
        visual_q: queue.Queue = queue.Queue()

        with self._pending_lock:
            self._pending[corr_id] = {"text": text_q, "visual": visual_q}

        req = schemas.retrieve_request(corr_id, question, TOP_K)
        self._producer.send(schemas.TOPIC_RETRIEVE_TEXT_REQ, req)
        self._producer.send(schemas.TOPIC_RETRIEVE_VIS_REQ,  req)
        self._producer.flush()

        try:
            text_hits   = text_q.get(timeout=RETRIEVE_TIMEOUT)
            visual_hits = visual_q.get(timeout=RETRIEVE_TIMEOUT)
        except queue.Empty:
            log.warning("Retrieval timeout for %s", corr_id)
            text_hits, visual_hits = [], []
        finally:
            with self._pending_lock:
                self._pending.pop(corr_id, None)

        best_score = max(
            [h["score"] for h in text_hits]
            + [h["score"] for h in visual_hits]
            + [0.0]
        )

        if best_score < SIMILARITY_THRESHOLD:
            return {
                "answer":            NO_INFO_MSG,
                "retrieved_chunks":  [],
                "visual_results":    [],
                "best_score":        best_score,
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
