"""Standalone query client — sends questions to the running RAG services via Redis.

Requires the full stack to be up (redis, text_indexer, visual_indexer, ollama).
Does NOT touch the indexing pipeline or flush any queues.

Usage:
  # Single question
  docker compose run --rm orchestrator python query.py --question "What is Fig. 4?"

  # Interactive REPL
  docker compose run --rm -it orchestrator python query.py
"""
import json
import logging
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import redis
from openai import OpenAI

sys.path.insert(0, "/app")
import shared.schemas as schemas

logging.basicConfig(level=logging.INFO, format="%(asctime)s [query] %(message)s")
log = logging.getLogger(__name__)

REDIS_URL            = os.environ.get("REDIS_URL",            "redis://redis:6379/0")
LLM_BASE_URL         = os.environ.get("LLM_BASE_URL",         "http://ollama:11434/v1")
LLM_MODEL            = os.environ.get("LLM_MODEL",            "gemma3:4b")
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.3"))
TOP_K                = int(os.environ.get("TOP_K",             "3"))
RETRIEVE_TIMEOUT     = int(os.environ.get("RETRIEVE_TIMEOUT",  "60"))
RETRIEVAL_LOG        = os.environ.get("RETRIEVAL_LOG",         "/data/logs/retrieval_log.jsonl")

NO_INFO_MSG = "The document does not contain information about this query."


def _save_images(visual_hits: list, corr_id: str) -> list[str]:
    """Decode base64 page images from visual hits and save them as PNG files.
    Returns a list of saved file paths."""
    import base64
    img_dir = Path(RETRIEVAL_LOG).parent / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, hit in enumerate(visual_hits):
        b64 = hit.get("base64")
        if not b64:
            continue
        page = hit.get("page_num", i)
        doc  = hit.get("doc_id", "unknown")
        path = img_dir / f"{corr_id[:8]}_doc{doc}_page{page}.png"
        path.write_bytes(base64.b64decode(b64))
        paths.append(str(path))
    return paths


def _append_log(question: str, result: dict) -> None:
    """Append one JSON line to the retrieval log file."""
    try:
        log_path = Path(RETRIEVAL_LOG)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": question,
            "answer": result.get("answer", ""),
            "best_score": result["best_score"],
            "chunks": result["retrieved_chunks"],
            "images": result.get("images", []),
        }
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        log.warning("Failed to write retrieval log", exc_info=True)


def query(question: str) -> dict:
    r   = redis.from_url(REDIS_URL, decode_responses=True)
    llm = OpenAI(base_url=LLM_BASE_URL, api_key="none")

    corr_id      = str(uuid.uuid4())
    text_res_q   = f"res.text.{corr_id}"
    visual_res_q = f"res.visual.{corr_id}"
    log.info("Query [%s]: %s", corr_id[:8], question[:80])

    text_req   = schemas.retrieve_request(corr_id, question, TOP_K)
    visual_req = schemas.retrieve_request(corr_id, question, TOP_K)
    text_req["reply_to"]   = text_res_q
    visual_req["reply_to"] = visual_res_q

    schemas.push(r, schemas.Q_RETRIEVE_TEXT_REQ, text_req)
    schemas.push(r, schemas.Q_RETRIEVE_VIS_REQ,  visual_req)
    log.info("Retrieve requests pushed, waiting for results …")

    text_hits:   list = []
    visual_hits: list = []

    def _wait_text():
        rc = redis.from_url(REDIS_URL, decode_responses=True)
        res = rc.brpop(text_res_q, timeout=RETRIEVE_TIMEOUT)
        if res:
            text_hits.extend(json.loads(res[1])["hits"])
            log.info("Text results received (%d hits)", len(text_hits))
        else:
            log.warning("Text retrieval timeout")

    def _wait_visual():
        rc = redis.from_url(REDIS_URL, decode_responses=True)
        res = rc.brpop(visual_res_q, timeout=RETRIEVE_TIMEOUT)
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

    r.delete(text_res_q, visual_res_q)

    best_text_score = max([h["score"] for h in text_hits] + [0.0])
    best_score = max([h["score"] for h in text_hits] + [h["score"] for h in visual_hits] + [0.0])

    # Gate on text score only — visual (ColQwen2 MaxSim) scores are 5-20+ regardless of relevance
    if best_text_score < SIMILARITY_THRESHOLD:
        result = {
            "answer":           NO_INFO_MSG,
            "retrieved_chunks": [],
            "visual_results":   [],
            "best_score":       best_score,
            "images":           [],
        }
        _append_log(question, result)
        return result

    # Build numbered text chunks with source labels
    text_sections = "\n\n".join(
        f"[Text {i+1} | pdf:{h['pdf_id']} | score:{h['score']:.3f}]\n{h['text']}"
        for i, h in enumerate(text_hits)
    )

    system_msg = (
        "You are a scientific document assistant. "
        "Answer the user's question using ALL of the provided context: "
        "text excerpts (which may include tables in Markdown), and page images "
        "(which may contain figures, chemical schemes, and visual diagrams). "
        "If the answer is visible in an image but not in the text, describe what you see. "
        "Be specific and concise."
    )

    user_content: list[dict] = [
        {"type": "text", "text": f"Text context:\n\n{text_sections}\n\nQuestion: {question}"},
    ]
    for hit in visual_hits:
        b64 = hit.get("base64")
        if b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=512,
    )
    images = _save_images(visual_hits, corr_id)
    result = {
        "answer": resp.choices[0].message.content.strip(),
        "retrieved_chunks": [
            {"score": h["score"], "text": h["text"], "pdf_id": h["pdf_id"]}
            for h in text_hits
        ],
        "visual_results": [
            {"score": h["score"], "doc_id": h.get("doc_id"), "page_num": h.get("page_num")}
            for h in visual_hits
        ],
        "best_score": best_score,
        "images":     images,
    }
    _append_log(question, result)
    return result


def _print_result(result: dict) -> None:
    print(f"Answer : {result['answer']}")
    print(f"Score  : {result['best_score']:.3f}")
    for c in result["retrieved_chunks"]:
        print(f"  [{c['score']:.3f}] ({c['pdf_id']}) {c['text']}")
    for path in result.get("images", []):
        print(f"  [image] {path}")
    print()


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="RAG query client")
    ap.add_argument("--question", default=None, help="Single question (omit for interactive REPL)")
    args = ap.parse_args()

    if args.question:
        _print_result(query(args.question))
    else:
        print("Interactive mode — type a question or 'quit' to exit.")
        for line in sys.stdin:
            q = line.strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                _print_result(query(q))


if __name__ == "__main__":
    _cli()
