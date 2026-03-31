"""Part 3: Agentic Orchestrator — RAG service as a native OpenAI tool.

The agent calls the Part 2 Docker RAG service (via Redis) only when the
question requires document knowledge; otherwise it answers directly.

Requires the Part 2 Docker stack to be running and already indexed
(redis, text_indexer, visual_indexer, ollama).

Usage:
    python -m src.part3.agent
    python -m src.part3.agent --question "What does Fig. 4 show?"

Environment variables (defaults target localhost):
    REDIS_URL, LLM_BASE_URL, LLM_MODEL, SIMILARITY_THRESHOLD, TOP_K, RETRIEVE_TIMEOUT
"""
from __future__ import annotations

import json
import os
import sys
import threading
import uuid

import redis
from openai import OpenAI

import src.part2.shared.schemas as schemas

REDIS_URL            = os.environ.get("REDIS_URL",            "redis://localhost:6379/0")
LLM_BASE_URL         = os.environ.get("LLM_BASE_URL",         "http://localhost:11434/v1")
LLM_MODEL            = os.environ.get("LLM_MODEL",            "gemma3:4b")
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.3"))
TOP_K                = int(os.environ.get("TOP_K",             "3"))
RETRIEVE_TIMEOUT     = int(os.environ.get("RETRIEVE_TIMEOUT",  "60"))

NO_INFO_MSG = "The document does not contain information about this query."


# ── Tool implementation ────────────────────────────────────────────────────────

def _trace(msg: str) -> None:
    print(f"  \033[36m[trace]\033[0m {msg}", flush=True)


def rag_query(question: str) -> str:
    """Query the Part 2 RAG services via Redis and return a generated answer."""
    r = redis.from_url(REDIS_URL, decode_responses=True)
    corr_id      = str(uuid.uuid4())
    text_res_q   = f"res.text.{corr_id}"
    visual_res_q = f"res.visual.{corr_id}"

    _trace(f"tool=rag_query  corr_id={corr_id[:8]}")
    _trace(f"pushing to queues: {schemas.Q_RETRIEVE_TEXT_REQ}, {schemas.Q_RETRIEVE_VIS_REQ}")

    text_req = {**schemas.retrieve_request(corr_id, question, TOP_K), "reply_to": text_res_q}
    vis_req  = {**schemas.retrieve_request(corr_id, question, TOP_K), "reply_to": visual_res_q}
    schemas.push(r, schemas.Q_RETRIEVE_TEXT_REQ, text_req)
    schemas.push(r, schemas.Q_RETRIEVE_VIS_REQ,  vis_req)

    text_hits: list = []
    visual_hits: list = []

    def _wait(queue: str, out: list) -> None:
        rc = redis.from_url(REDIS_URL, decode_responses=True)
        res = rc.brpop(queue, timeout=RETRIEVE_TIMEOUT)
        if res:
            out.extend(json.loads(res[1])["hits"])

    t1 = threading.Thread(target=_wait, args=(text_res_q,   text_hits))
    t2 = threading.Thread(target=_wait, args=(visual_res_q, visual_hits))
    t1.start(); t2.start()
    t1.join(timeout=RETRIEVE_TIMEOUT + 5)
    t2.join(timeout=RETRIEVE_TIMEOUT + 5)
    r.delete(text_res_q, visual_res_q)

    best_text = max([h["score"] for h in text_hits] + [0.0])
    _trace(f"text hits: {len(text_hits)}  best_score: {best_text:.3f}  threshold: {SIMILARITY_THRESHOLD}")
    for h in text_hits:
        _trace(f"  text  score={h['score']:.3f}  pdf={h['pdf_id']}  \"{h['text'][:80].strip()}…\"")
    _trace(f"visual hits: {len(visual_hits)}")
    for h in visual_hits:
        _trace(f"  visual  score={h['score']:.3f}  doc={h.get('doc_id')}  page={h.get('page_num')}  has_image={'yes' if h.get('base64') else 'no'}")

    if best_text < SIMILARITY_THRESHOLD:
        _trace("below threshold → returning NO_INFO_MSG (no LLM call)")
        return NO_INFO_MSG

    text_ctx = "\n\n".join(
        f"[score:{h['score']:.3f} | pdf:{h['pdf_id']}]\n{h['text']}" for h in text_hits
    )
    user_content: list[dict] = [
        {"type": "text", "text": f"Text context:\n\n{text_ctx}\n\nQuestion: {question}"},
    ]
    for hit in visual_hits:
        b64 = hit.get("base64")
        if b64:
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    _trace(f"calling LLM ({LLM_MODEL}) with {len(text_hits)} text chunks + {sum(1 for h in visual_hits if h.get('base64'))} images")
    llm = OpenAI(base_url=LLM_BASE_URL, api_key="none")
    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a scientific document assistant. Answer using the provided context."},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


# ── Agent loop ────────────────────────────────────────────────────────────────

_ROUTER_PROMPT = (
    "You are a router. Decide if the question requires searching scientific PDF documents "
    "or can be answered from general knowledge.\n"
    "Reply with exactly one word: YES (needs documents) or NO (general knowledge)."
)


def run_agent(question: str) -> str:
    """Two-step agentic loop: classify → (optionally) call RAG tool → respond."""
    llm = OpenAI(base_url=LLM_BASE_URL, api_key="none")

    # Step 1: route — does this need the document index?
    route = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _ROUTER_PROMPT},
            {"role": "user",   "content": question},
        ],
        max_tokens=5,
    ).choices[0].message.content.strip().upper()

    needs_rag = route.startswith("YES")
    _trace(f"router decision: {route!r} → {'rag_query' if needs_rag else 'direct answer'}")

    if needs_rag:
        rag_result = rag_query(question)
        _trace("synthesising final answer from RAG result")
        # Step 2a: synthesise RAG result into a final answer
        return llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer the question using the provided context."},
                {"role": "user",   "content": f"Context:\n{rag_result}\n\nQuestion: {question}"},
            ],
            max_tokens=512,
        ).choices[0].message.content.strip()

    # Step 2b: answer directly from general knowledge
    _trace("answering directly (no retrieval)")
    return llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": question},
        ],
        max_tokens=512,
    ).choices[0].message.content.strip()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Part 3 agentic RAG orchestrator")
    ap.add_argument("--question", default=None, help="Single question (omit for interactive REPL)")
    args = ap.parse_args()

    def _ask(q: str) -> None:
        print(f"\nAnswer: {run_agent(q)}\n")

    if args.question:
        _ask(args.question)
    else:
        print("Agent ready. Type a question or 'quit' to exit.")
        for line in sys.stdin:
            q = line.strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                _ask(q)


if __name__ == "__main__":
    _cli()
