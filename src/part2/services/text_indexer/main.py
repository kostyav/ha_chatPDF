"""Text indexer service — two concurrent Kafka loops:
  1. index_loop   : consumes parse.results → upserts chunks into Qdrant → produces index.ready
  2. retrieve_loop: consumes retrieve.text.requests → queries Qdrant → produces retrieve.text.results

Persistence: vectors live in Qdrant backed by a Docker volume, so restarting
the stack does NOT re-index PDFs that are already present in the collection.
Each chunk is identified by a deterministic UUID derived from (pdf_id, text),
making upserts idempotent.
"""
import hashlib
import json
import logging
import os
import re
import sys
import threading
import uuid

import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams,
)
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "/app")
import shared.schemas as schemas

logging.basicConfig(level=logging.INFO, format="%(asctime)s [text_indexer] %(message)s")
log = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092")
QDRANT_HOST     = os.environ.get("QDRANT_HOST",     "qdrant")
QDRANT_PORT     = int(os.environ.get("QDRANT_PORT", "6333"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_MAX_CHARS = int(os.environ.get("CHUNK_MAX_CHARS", "800"))
COLLECTION      = "rag_chunks"


# ── Chunking ───────────────────────────────────────────────────────────────────

def _split_markdown(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
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


def _chunk_id(pdf_id: str, text: str) -> str:
    """Deterministic UUID — same chunk always maps to the same Qdrant point ID."""
    return str(uuid.UUID(hashlib.md5(f"{pdf_id}:{text}".encode()).hexdigest()))


# ── Qdrant-backed index ────────────────────────────────────────────────────────

class _TextIndex:
    def __init__(self, model_name: str, qdrant: QdrantClient) -> None:
        self.model  = SentenceTransformer(model_name)
        self.qdrant = qdrant
        vector_size = self.model.get_sentence_embedding_dimension()
        self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        existing = {c.name for c in self.qdrant.get_collections().collections}
        if COLLECTION not in existing:
            self.qdrant.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            log.info("Created Qdrant collection '%s' (dim=%d)", COLLECTION, vector_size)
        else:
            log.info("Qdrant collection '%s' already exists — reusing", COLLECTION)

    def _is_indexed(self, pdf_id: str) -> bool:
        results, _ = self.qdrant.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="pdf_id", match=MatchValue(value=pdf_id))
            ]),
            limit=1,
        )
        return len(results) > 0

    def add(self, pdf_id: str, markdown: str, tables_md: list[str]) -> int:
        if self._is_indexed(pdf_id):
            log.info("Skipping %s — already indexed in Qdrant", pdf_id)
            return 0

        full  = markdown + ("\n\n" + "\n\n".join(tables_md) if tables_md else "")
        texts = _split_markdown(full)
        embs  = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

        points = [
            PointStruct(
                id=_chunk_id(pdf_id, text),
                vector=emb.tolist(),
                payload={"text": text, "pdf_id": pdf_id},
            )
            for text, emb in zip(texts, embs)
        ]
        self.qdrant.upsert(collection_name=COLLECTION, points=points)
        log.info("Upserted %d chunks for %s", len(points), pdf_id)
        return len(points)

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        results = self.qdrant.search(
            collection_name=COLLECTION,
            query_vector=q[0].tolist(),
            limit=k,
            with_payload=True,
        )
        return [
            {"score": r.score, "text": r.payload["text"], "pdf_id": r.payload["pdf_id"]}
            for r in results
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
        log.info("%s: %s", data["pdf_id"],
                 f"indexed {n} new chunks" if n > 0 else "skipped (already in Qdrant)")


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
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    idx    = _TextIndex(EMBEDDING_MODEL, qdrant)

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
    log.info("Text indexer started (Qdrant at %s:%d)", QDRANT_HOST, QDRANT_PORT)
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
