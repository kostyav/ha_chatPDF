"""Kafka topic names and JSON message schemas for inter-service communication.

Every Kafka message is a JSON-serialised dict.  The constructors below document
the exact shape expected by each consumer; add/remove fields here and update the
corresponding service to keep both sides in sync.
"""

# ── Topic names ────────────────────────────────────────────────────────────────

# Indexing pipeline
TOPIC_PARSE_REQUESTS = "parse.requests"   # orchestrator → parser, visual_indexer
TOPIC_PARSE_RESULTS  = "parse.results"    # parser        → text_indexer
TOPIC_INDEX_READY    = "index.ready"      # text_indexer, visual_indexer → orchestrator

# Query pipeline
TOPIC_RETRIEVE_TEXT_REQ = "retrieve.text.requests"    # orchestrator  → text_indexer
TOPIC_RETRIEVE_TEXT_RES = "retrieve.text.results"     # text_indexer  → orchestrator
TOPIC_RETRIEVE_VIS_REQ  = "retrieve.visual.requests"  # orchestrator  → visual_indexer
TOPIC_RETRIEVE_VIS_RES  = "retrieve.visual.results"   # visual_indexer → orchestrator

ALL_TOPICS = [
    TOPIC_PARSE_REQUESTS,
    TOPIC_PARSE_RESULTS,
    TOPIC_INDEX_READY,
    TOPIC_RETRIEVE_TEXT_REQ,
    TOPIC_RETRIEVE_TEXT_RES,
    TOPIC_RETRIEVE_VIS_REQ,
    TOPIC_RETRIEVE_VIS_RES,
]


# ── Message constructors ───────────────────────────────────────────────────────

def parse_request(pdf_path: str, pdf_id: str) -> dict:
    """orchestrator → parser, visual_indexer"""
    return {"pdf_path": pdf_path, "pdf_id": pdf_id}


def parse_result(pdf_id: str, markdown: str, tables_md: list[str]) -> dict:
    """parser → text_indexer"""
    return {"pdf_id": pdf_id, "markdown": markdown, "tables_md": tables_md}


def index_ready(service: str, pdf_id: str) -> dict:
    """text_indexer | visual_indexer → orchestrator
    service: "text" or "visual"
    """
    return {"service": service, "pdf_id": pdf_id}


def retrieve_request(correlation_id: str, question: str, top_k: int = 3) -> dict:
    """orchestrator → text_indexer | visual_indexer"""
    return {"correlation_id": correlation_id, "question": question, "top_k": top_k}


def retrieve_text_result(correlation_id: str, hits: list[dict]) -> dict:
    """text_indexer → orchestrator
    hits: [{"score": float, "text": str, "pdf_id": str}]
    """
    return {"correlation_id": correlation_id, "hits": hits}


def retrieve_visual_result(correlation_id: str, hits: list[dict]) -> dict:
    """visual_indexer → orchestrator
    hits: [{"score": float, "doc_id": any, "page_num": any, "base64": str | None}]
    """
    return {"correlation_id": correlation_id, "hits": hits}
