"""Visual indexer service — two concurrent Kafka loops:
  1. index_loop   : consumes parse.requests → indexes PDF via Byaldi → produces index.ready
  2. retrieve_loop: consumes retrieve.visual.requests → queries Byaldi → produces retrieve.visual.results

Dependency note: this container pins transformers>=4.47,<4.48 to satisfy
colpali-engine.  Docling's rt_detr_v2 layout model requires >=4.48, which is
why the parser lives in its own container.

The visual indexer consumes parse.REQUESTS (not results) because Byaldi indexes
PDF files directly — it does not need the Docling-extracted text.
"""
import json
import logging
import os
import sys
import threading

from kafka import KafkaConsumer, KafkaProducer

sys.path.insert(0, "/app")
import shared.schemas as schemas

logging.basicConfig(level=logging.INFO, format="%(asctime)s [visual_indexer] %(message)s")
log = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092")
COLQWEN_MODEL   = os.environ.get("COLQWEN_MODEL", "vidore/colqwen2-v0.1")


# ── Byaldi wrapper ─────────────────────────────────────────────────────────────

class _VisualIndex:
    def __init__(self):
        self._model        = None
        self._indexed_count = 0
        self._lock         = threading.Lock()

    def _get_model(self):
        if self._model is None:
            from byaldi import RAGMultiModalModel
            self._model = RAGMultiModalModel.from_pretrained(COLQWEN_MODEL, verbose=0)
        return self._model

    def add(self, pdf_path: str) -> None:
        with self._lock:
            model = self._get_model()
            if self._indexed_count == 0:
                model.index(pdf_path, index_name="visual_index", overwrite=True)
            else:
                model.add_to_index(pdf_path, store_collection_with_index=False)
            self._indexed_count += 1

    def search(self, query: str, k: int = 3) -> list[dict]:
        with self._lock:
            if self._model is None or self._indexed_count == 0:
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
            idx.add(req["pdf_path"])
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
    log.info("Visual indexer service started")
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
