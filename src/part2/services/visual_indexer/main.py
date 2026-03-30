"""Visual indexer service — two concurrent Kafka loops:
  1. index_loop   : consumes parse.requests → indexes PDF via Byaldi → produces index.ready
  2. retrieve_loop: consumes retrieve.visual.requests → queries Byaldi → produces retrieve.visual.results

Persistence across restarts:
  - A JSON state file (INDEX_DIR/indexed_pdfs.json) tracks which pdf_ids are stored.
  - On startup, if the state file is non-empty the existing Byaldi index is loaded
    from disk instead of rebuilt — avoiding a full re-index on every docker compose up.
  - PDFs already present in the state file are skipped in index_loop.
"""
import json
import logging
import os
import sys
import threading
from pathlib import Path

from kafka import KafkaConsumer, KafkaProducer

sys.path.insert(0, "/app")
import shared.schemas as schemas

logging.basicConfig(level=logging.INFO, format="%(asctime)s [visual_indexer] %(message)s")
log = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092")
COLQWEN_MODEL   = os.environ.get("COLQWEN_MODEL",   "vidore/colqwen2-v0.1")
INDEX_DIR       = Path(os.environ.get("INDEX_DIR",  "/data/byaldi_index"))
INDEX_NAME      = "visual_index"
STATE_FILE      = INDEX_DIR / "indexed_pdfs.json"


# ── Byaldi wrapper with persistence ───────────────────────────────────────────

class _VisualIndex:
    def __init__(self) -> None:
        self._model             = None
        self._indexed: set[str] = self._load_state()
        # True when we loaded an existing index from disk
        self._has_existing      = bool(self._indexed)
        self._lock              = threading.Lock()

    # ── State file ────────────────────────────────────────────────────────────

    def _load_state(self) -> set[str]:
        if STATE_FILE.exists():
            try:
                return set(json.loads(STATE_FILE.read_text()))
            except Exception:
                log.warning("Could not read state file — starting fresh")
        return set()

    def _save_state(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(sorted(self._indexed)))

    # ── Byaldi model ──────────────────────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            from byaldi import RAGMultiModalModel
            if self._has_existing:
                log.info("Loading existing Byaldi index from %s …", INDEX_DIR)
                self._model = RAGMultiModalModel.from_index(
                    INDEX_NAME,
                    index_root=str(INDEX_DIR),
                    verbose=0,
                )
            else:
                self._model = RAGMultiModalModel.from_pretrained(
                    COLQWEN_MODEL, verbose=0
                )
        return self._model

    # ── Public interface ──────────────────────────────────────────────────────

    def add(self, pdf_path: str, pdf_id: str) -> None:
        with self._lock:
            if pdf_id in self._indexed:
                log.info("Skipping %s — already in visual index", pdf_id)
                return
            model = self._get_model()
            if self._has_existing or self._indexed:
                # Existing index loaded from disk, or at least one PDF already indexed
                model.add_to_index(pdf_path, store_collection_with_index=False)
            else:
                # Very first PDF — create the index (no overwrite of persisted data)
                model.index(
                    pdf_path,
                    index_name=INDEX_NAME,
                    index_root=str(INDEX_DIR),
                    overwrite=False,
                )
            self._indexed.add(pdf_id)
            self._save_state()

    def search(self, query: str, k: int = 3) -> list[dict]:
        with self._lock:
            if self._model is None or not self._indexed:
                return []
            results = self._model.search(query, k=k, return_base64_results=True)
        return [
            {
                "score":    float(r.score),
                "doc_id":   getattr(r, "doc_id",  None),
                "page_num": getattr(r, "page_num", None),
                "base64":   getattr(r, "base64",  None),
            }
            for r in results
        ]


# ── Consumer loops ─────────────────────────────────────────────────────────────

def _index_loop(idx: _VisualIndex, producer: KafkaProducer) -> None:
    consumer = KafkaConsumer(
        schemas.TOPIC_PARSE_REQUESTS,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="visual_indexer",
        value_deserializer=lambda b: json.loads(b.decode()),
        auto_offset_reset="earliest",
    )
    log.info("index_loop ready")
    for msg in consumer:
        req = msg.value
        try:
            idx.add(req["pdf_path"], req["pdf_id"])
            producer.send(schemas.TOPIC_INDEX_READY, schemas.index_ready("visual", req["pdf_id"]))
            producer.flush()
            log.info("Visual-indexed %s", req["pdf_id"])
        except Exception:
            log.exception("Failed to visual-index %s", req["pdf_id"])


def _retrieve_loop(idx: _VisualIndex, producer: KafkaProducer) -> None:
    consumer = KafkaConsumer(
        schemas.TOPIC_RETRIEVE_VIS_REQ,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="visual_indexer_retrieve",
        value_deserializer=lambda b: json.loads(b.decode()),
        auto_offset_reset="latest",
    )
    log.info("retrieve_loop ready")
    for msg in consumer:
        req  = msg.value
        hits = idx.search(req["question"], k=req.get("top_k", 3))
        producer.send(
            schemas.TOPIC_RETRIEVE_VIS_RES,
            schemas.retrieve_visual_result(req["correlation_id"], hits),
        )
        producer.flush()


def main() -> None:
    idx = _VisualIndex()
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode(),
    )
    threads = [
        threading.Thread(target=_index_loop,    args=(idx, producer), daemon=True),
        threading.Thread(target=_retrieve_loop, args=(idx, producer), daemon=True),
    ]
    for t in threads:
        t.start()
    log.info("Visual indexer started (index_dir=%s, cached=%s)",
             INDEX_DIR, sorted(idx._indexed) or "none")
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
