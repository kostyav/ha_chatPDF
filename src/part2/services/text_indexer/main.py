"""Text indexer service — two concurrent Kafka loops:
  1. index_loop   : consumes parse.results → builds FAISS index → produces index.ready
  2. retrieve_loop: consumes retrieve.text.requests → queries FAISS → produces retrieve.text.results
"""
import json
import logging
import os
import re
import sys
import threading
from pathlib import Path

import faiss
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "/app")
import shared.schemas as schemas

logging.basicConfig(level=logging.INFO, format="%(asctime)s [text_indexer] %(message)s")
log = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_MAX_CHARS = int(os.environ.get("CHUNK_MAX_CHARS", "800"))


# ── Chunking ───────────────────────────────────────────────────────────────────

def _split_markdown(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """Split markdown on section headers, then by paragraph if a section is too large."""
    sections = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)
    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= max_chars:
            chunks.append(section)
        else:
            paragraphs = re.split(r"\n\n+", section)
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


# ── In-memory index (shared between both threads) ─────────────────────────────

class _TextIndex:
    def __init__(self, model_name: str):
        self.model  = SentenceTransformer(model_name)
        self.chunks: list[dict] = []
        self.index: faiss.Index | None = None
        self._lock  = threading.Lock()

    def add(self, pdf_id: str, markdown: str, tables_md: list[str]) -> int:
        full = markdown + ("\n\n" + "\n\n".join(tables_md) if tables_md else "")
        texts = _split_markdown(full)
        embs  = self.model.encode(texts, normalize_embeddings=True,
                                  show_progress_bar=False).astype(np.float32)
        with self._lock:
            if self.index is None:
                self.index = faiss.IndexFlatIP(embs.shape[1])
            self.index.add(embs)
            self.chunks.extend({"text": t, "pdf_id": pdf_id} for t in texts)
        return len(texts)

    def search(self, query: str, k: int = 3) -> list[dict]:
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            k = min(k, self.index.ntotal)
            q = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
            scores, ids = self.index.search(q, k)
        return [
            {"score": float(scores[0][i]),
             "text":  self.chunks[ids[0][i]]["text"],
             "pdf_id": self.chunks[ids[0][i]]["pdf_id"]}
            for i in range(k) if ids[0][i] >= 0
        ]


# ── Consumer loops ─────────────────────────────────────────────────────────────

def _index_loop(idx: _TextIndex, producer: KafkaProducer) -> None:
    consumer = KafkaConsumer(
        schemas.TOPIC_PARSE_RESULTS,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="text_indexer",
        value_deserializer=lambda b: json.loads(b.decode()),
        auto_offset_reset="earliest",
    )
    log.info("index_loop ready")
    for msg in consumer:
        data = msg.value
        n = idx.add(data["pdf_id"], data["markdown"], data["tables_md"])
        producer.send(schemas.TOPIC_INDEX_READY, schemas.index_ready("text", data["pdf_id"]))
        producer.flush()
        log.info("Indexed %d chunks for %s", n, data["pdf_id"])


def _retrieve_loop(idx: _TextIndex, producer: KafkaProducer) -> None:
    consumer = KafkaConsumer(
        schemas.TOPIC_RETRIEVE_TEXT_REQ,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="text_indexer_retrieve",
        value_deserializer=lambda b: json.loads(b.decode()),
        auto_offset_reset="latest",
    )
    log.info("retrieve_loop ready")
    for msg in consumer:
        req  = msg.value
        hits = idx.search(req["question"], k=req.get("top_k", 3))
        producer.send(
            schemas.TOPIC_RETRIEVE_TEXT_RES,
            schemas.retrieve_text_result(req["correlation_id"], hits),
        )
        producer.flush()


def main() -> None:
    idx = _TextIndex(EMBEDDING_MODEL)
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
    log.info("Text indexer service started")
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
