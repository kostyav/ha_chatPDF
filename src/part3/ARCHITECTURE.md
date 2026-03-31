# Architecture — Part 3: Agentic Orchestrator

## Overview

Part 3 wraps the Part 2 RAG system as a callable tool inside an autonomous agent.
The agent evaluates each user question and decides whether to retrieve information
from the scientific document index or answer directly from general knowledge.

The agent does **not** index documents — it is a pure query-time client that
dispatches to the already-running Part 2 Docker services.

---

## File Layout

```
src/part3/
├── ARCHITECTURE.md       ← this file
├── README.md             ← task spec + run instructions
├── __init__.py
├── agent.py              ← agent loop + RAG tool + trace (single file, ~190 LOC)
│     ├── _trace()            prints cyan [trace] lines at every decision point
│     ├── rag_query()         the tool: Redis → retrieve → multimodal LLM answer
│     ├── run_agent()         two-step routing loop (router LLM call → dispatch)
│     └── _cli()              argparse entry point; interactive REPL or single question
├── Dockerfile            ← python:3.12-slim image; copies only the needed src/
└── docker-compose.yml    ← joins rag-part2_rag-net; sets env vars
```

---

## Design

### Why no LangChain / LangGraph?

`gemma3:4b` (the only model available in the Docker stack) does not support the
OpenAI `tools` API parameter. Native tool-calling frameworks (LangChain's
`create_tool_calling_agent`, LangGraph, etc.) all rely on that API and would
fail with a `400 does not support tools` error.

The solution is a **two-step prompt-based routing loop** that works with any
instruction-following model — no tool-calling support required.

---

## Agent Decision Flow

```
User question
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Step 1 — Router (LLM call, max_tokens=5)           │
│                                                     │
│  Prompt: "Does this require searching scientific    │
│           PDF documents? Reply YES or NO."          │
│                                                     │
│  Response: "YES" or "NO"                            │
└──────────────┬──────────────────────────────────────┘
               │
       ┌───────┴────────┐
       │ YES             │ NO
       ▼                 ▼
┌─────────────┐    ┌──────────────────────────────────┐
│  rag_query  │    │  Step 2b — Direct LLM answer     │
│  (see below)│    │  (general knowledge, no retrieval)│
└──────┬──────┘    └──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Step 2a — Synthesis (LLM call)                     │
│  Prompt: RAG result as context + original question  │
│  Response: grounded final answer                    │
└─────────────────────────────────────────────────────┘
```

---

## RAG Tool — `rag_query(question)`

Calls the **already-running Part 2 microservices** via Redis.  The flow mirrors
`src/part2/services/orchestrator/query.py` exactly:

```
rag_query(question)
  │
  │  LPUSH retrieve.text.requests   {corr_id, question, top_k}
  ├──────────────────────────────────────────► text_indexer  (Qdrant HNSW)
  │                                              │ LPUSH res.text.<corr_id>  {hits}
  │                                              └──────────────────────────► agent
  │
  │  LPUSH retrieve.visual.requests {corr_id, question, top_k}
  └──────────────────────────────────────────► visual_indexer (ColQwen2/Byaldi)
                                                 │ LPUSH res.visual.<corr_id> {hits}
                                                 └──────────────────────────► agent

  agent: collects both result queues (two threads, BRPOP with timeout)
       → applies similarity threshold on text score
       → builds multimodal prompt (text chunks + base64 page images)
       → calls Ollama → returns generated answer string
```

Correlation IDs (`uuid4`) keep per-query reply queues isolated.
Both Redis lists are deleted after receipt.

---

## Interaction Trace

Each query prints a `[trace]` line (cyan) at every decision point so tool usage
is visible to the user:

```
Agent ready. Type a question or 'quit' to exit.

> Short description of Bhutan
  [trace] router decision: 'NO' → direct answer
  [trace] answering directly (no retrieval)

Answer: Bhutan is a landlocked country in the Eastern Himalayas …

> What does Fig. 4 show?
  [trace] router decision: 'YES' → rag_query
  [trace] tool=rag_query  corr_id=f3a1b2c4
  [trace] pushing to queues: retrieve.text.requests, retrieve.visual.requests
  [trace] text hits: 3  best_score: 0.651  threshold: 0.3
  [trace]   text  score=0.651  pdf=23870758  "## Figure 4  The training loss curves for…"
  [trace]   text  score=0.490  pdf=23870758  "Table 2 summarises the convergence behavi…"
  [trace]   text  score=0.371  pdf=23870758  "We compare the proposed model against thre…"
  [trace] visual hits: 2
  [trace]   visual  score=14.3  doc=0  page=3  has_image=yes
  [trace]   visual  score=11.8  doc=0  page=5  has_image=yes
  [trace] calling LLM (gemma3:4b) with 3 text chunks + 2 images
  [trace] synthesising final answer from RAG result

Answer: Figure 4 shows the training loss curves comparing the baseline and the
        proposed model over 100 epochs …
```

The trace shows exactly which branch was taken (router), what Redis returned
(hit counts, scores, source PDFs, page images), and which LLM calls were made.

---

## LLM Calls Per Query

| Step | Purpose | max_tokens |
|------|---------|-----------|
| Router | YES/NO routing decision | 5 |
| RAG synthesis (if YES) | Generate grounded answer from retrieved context | 512 |
| Direct answer (if NO) | Answer from general knowledge | 512 |

All three calls go to the same Ollama endpoint (`gemma3:4b`).
If the router says YES, there are **two** LLM calls total (router + synthesis)
plus the internal generation call inside `rag_query` — three in total.

---

## Network Architecture

```
Host machine
  └── Docker engine
        │
        ├── rag-part2_rag-net  (internal bridge network, no host port bindings)
        │     ├── redis:6379
        │     ├── ollama:11434
        │     ├── qdrant:6333
        │     ├── text_indexer
        │     ├── visual_indexer
        │     └── agent  ← Part 3 container (joins this network)
        │
        └── (host) — cannot reach redis/ollama directly
```

The Part 3 `docker-compose.yml` declares `rag-part2_rag-net` as an **external**
network so the agent container is placed on it at startup.

---

## Configuration

All parameters are environment variables with defaults suitable for the Docker environment:

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection string |
| `LLM_BASE_URL` | `http://ollama:11434/v1` | Ollama OpenAI-compatible endpoint |
| `LLM_MODEL` | `gemma3:4b` | Model used for routing + generation |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum text score to use RAG result |
| `TOP_K` | `3` | Retrieved chunks / images per query |
| `RETRIEVE_TIMEOUT` | `60` | Seconds to wait for retrieval results |

---

## Dependencies

Only two runtime dependencies beyond the standard library:

| Package | Version | Use |
|---------|---------|-----|
| `openai` | ≥1.0 | LLM calls (Ollama OpenAI-compatible API) |
| `redis` | ≥5.0 | Push/pop messages to Part 2 queues |

`src.part2.shared.schemas` is imported directly from the Part 2 package
(no duplication) and provides queue names and message constructors.
