"""Visual indexer service — two concurrent loops:
  1. index_loop   : consumes parse.requests → indexes PDF via Byaldi → produces index.ready
  2. retrieve_loop: consumes retrieve.visual.requests → queries Byaldi → produces retrieve.visual.results
"""
import json
import logging
import os
import sys
import threading
from pathlib import Path

import redis

sys.path.insert(0, "/app")
import shared.schemas as schemas

logging.basicConfig(level=logging.INFO, format="%(asctime)s [visual_indexer] %(message)s")
log = logging.getLogger(__name__)

REDIS_URL     = os.environ.get("REDIS_URL",    "redis://redis:6379/0")
COLQWEN_MODEL = os.environ.get("COLQWEN_MODEL", "vidore/colqwen2-v0.1")
INDEX_DIR     = Path(os.environ.get("INDEX_DIR", "/data/byaldi_index"))
INDEX_NAME    = "visual_index"
STATE_FILE    = INDEX_DIR / "indexed_pdfs.json"


# ── Byaldi wrapper with persistence ───────────────────────────────────────────

class _VisualIndex:
    def __init__(self) -> None:
        self._model             = None
        self._indexed: set[str] = self._load_state()
        self._has_existing      = bool(self._indexed)
        self._lock              = threading.Lock()

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

    def _get_model(self):
        if self._model is None:
            from byaldi import RAGMultiModalModel
            if self._has_existing:
                log.info("Loading existing Byaldi index from %s …", INDEX_DIR)
                self._model = RAGMultiModalModel.from_index(
                    INDEX_NAME, index_root=str(INDEX_DIR), verbose=0,
                )
            else:
                self._model = RAGMultiModalModel.from_pretrained(
                    COLQWEN_MODEL, index_root=str(INDEX_DIR), verbose=0,
                )
        return self._model

    def add(self, pdf_path: str, pdf_id: str) -> bool:
        """Returns True if indexing was performed, False if already cached."""
        with self._lock:
            if pdf_id in self._indexed:
                log.info("Skipping %s — already in visual index", pdf_id)
                return False
            model = self._get_model()
            if self._has_existing or self._indexed:
                model.add_to_index(pdf_path, store_collection_with_index=True)
            else:
                model.index(
                    pdf_path,
                    index_name=INDEX_NAME,
                    store_collection_with_index=True,
                    overwrite=False,
                )
            self._indexed.add(pdf_id)
            self._save_state()
            return True

    def search(self, query: str, k: int = 3) -> list[dict]:
        with self._lock:
            if not self._indexed:
                return []
            results = self._get_model().search(query, k=k, return_base64_results=True)
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

def _index_loop(idx: _VisualIndex, r: redis.Redis) -> None:
    log.info("index_loop ready")
    while True:
        req = schemas.pop(r, schemas.Q_PARSE_REQUESTS)
        if req is None:
            continue
        try:
            already_cached = not idx.add(req["pdf_path"], req["pdf_id"])
            schemas.push(r, schemas.Q_INDEX_READY,
                         schemas.index_ready("visual", req["pdf_id"]))
            if already_cached:
                log.info("Visual-index cache hit for %s — index.ready sent", req["pdf_id"])
            else:
                log.info("Visual-indexed %s", req["pdf_id"])
        except Exception:
            log.exception("Failed to visual-index %s", req["pdf_id"])


def _retrieve_loop(idx: _VisualIndex, r: redis.Redis) -> None:
    log.info("retrieve_loop ready")
    while True:
        req = schemas.pop(r, schemas.Q_RETRIEVE_VIS_REQ)
        if req is None:
            continue
        hits     = idx.search(req["question"], k=req.get("top_k", 3))
        reply_to = req.get("reply_to", schemas.Q_RETRIEVE_VIS_RES)
        schemas.push(r, reply_to,
                     schemas.retrieve_visual_result(req["correlation_id"], hits))


def main() -> None:
    r_main     = redis.from_url(REDIS_URL, decode_responses=True)
    r_index    = redis.from_url(REDIS_URL, decode_responses=True)
    r_retrieve = redis.from_url(REDIS_URL, decode_responses=True)
    r_main.delete(schemas.Q_PARSE_REQUESTS, schemas.Q_INDEX_READY,
                  schemas.Q_RETRIEVE_VIS_REQ, schemas.Q_RETRIEVE_VIS_RES)
    idx = _VisualIndex()
    threads = [
        threading.Thread(target=_index_loop,    args=(idx, r_index),    daemon=True),
        threading.Thread(target=_retrieve_loop, args=(idx, r_retrieve), daemon=True),
    ]
    for t in threads:
        t.start()
    log.info("Visual indexer started (index_dir=%s, cached=%s)",
             INDEX_DIR, sorted(idx._indexed) or "none")
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
