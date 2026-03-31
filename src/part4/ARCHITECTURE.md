# Part 4 — Architecture

## Overview

Part 4 is a thin HTTP layer over the Part 3 agentic flow. It adds no new logic: it wires FastAPI's async I/O and the OpenAI streaming API around the existing `run_agent` steps so the browser receives tokens as they are generated.

```
Browser
  │  GET /          → ui.html (single-page chat UI)
  │  POST /chat     → StreamingResponse (text/event-stream)
  ▼
main.py
  ├── _stream()          async generator — orchestrates the agent steps
  │     ├── AsyncOpenAI.chat.completions.create()   router call (step 1)
  │     ├── asyncio.to_thread(agent.rag_query)      RAG retrieval (step 2a, if needed)
  │     └── AsyncOpenAI.chat.completions.create(stream=True)  final answer (step 2b)
  │
  └── imports src.part3.agent  (rag_query, _ROUTER_PROMPT, LLM_BASE_URL, LLM_MODEL, …)
        └── imports src.part2.shared.schemas  (Redis queue names + push/pop helpers)
              └── Redis  (text_indexer + visual_indexer services from Part 2 stack)
```

---

## Modules

### `main.py`

The entire server in one file (~80 LOC).

| Symbol | Kind | Description |
|--------|------|-------------|
| `ChatRequest` | Pydantic model | Request body: `{"question": str}` |
| `_sse(event, data)` | helper function | Formats a string as an SSE frame: `event: …\ndata: …\n\n`. `data` is JSON-encoded so any character is safe to transmit. |
| `_stream(question)` | `async` generator | Core of the implementation. Runs the two-step agent logic and `yield`s SSE frames. See flow below. |
| `ui()` | FastAPI route `GET /` | Reads `ui.html` once at startup (`_UI`) and returns it as `HTMLResponse`. |
| `chat(req)` | FastAPI route `POST /chat` | Wraps `_stream` in a `StreamingResponse` with `media_type="text/event-stream"`. |

#### `_stream` — step-by-step flow

```
1. yield  status "Routing question…"
2. await  AsyncOpenAI — router prompt → YES / NO              (max_tokens=5, fast)
3a. if YES:
      yield  status "Retrieving from documents…"
      await  asyncio.to_thread(agent.rag_query, question)     (blocks in a thread — Redis + LLM)
      yield  status "Synthesizing answer…"
      build  messages = [system + "context + question"]
3b. if NO:
      yield  status "Answering from general knowledge…"
      build  messages = [system + question]
4. await  AsyncOpenAI(stream=True) → iterate chunks
      yield  token  <chunk.choices[0].delta.content>          (one yield per token)
5. yield  done ""
```

`asyncio.to_thread` is the key bridge: `agent.rag_query` uses blocking Redis `brpop` and the synchronous `openai.OpenAI` client internally; offloading it to a thread keeps the FastAPI event loop free.

---

### `ui.html`

Single-file chat interface (~80 LOC HTML/CSS/JS). No build step, no external assets.

| Section | Description |
|---------|-------------|
| CSS | Dark-themed layout; user bubbles right-aligned, agent bubbles left-aligned; `.status` italic label |
| `fetch("/chat", {method:"POST", …})` | Sends the question as JSON; receives the SSE stream via the `ReadableStream` API (no `EventSource` — `EventSource` only supports `GET`) |
| SSE parser | Splits the raw byte stream on `\n\n`, extracts `event:` and `data:` fields, `JSON.parse`s the payload |
| Event handlers | `status` → replaces the italic label above the agent bubble; `token` → appends to the agent bubble text; `done` → removes the status label |

---

### `Dockerfile`

Extends `python:3.12-slim`. Installs only the four runtime dependencies (`openai`, `redis`, `fastapi`, `uvicorn[standard]`). Copies `src/part2/shared/`, `src/part3/`, `src/part4/`, installs the local package (`pip install -e .`), and starts uvicorn on port 8080.

### `docker-compose.yml`

Single service `api`. Joins `rag-part2_rag-net` (the Docker network created by the Part 2 stack) so it can reach `redis` and `ollama` by hostname. Exposes port `8080` to the host.

---

## Dependency graph

```
src/part4/main.py
 └── src/part3/agent.py          (rag_query, _ROUTER_PROMPT, env-var constants)
      └── src/part2/shared/schemas.py   (Q_RETRIEVE_TEXT_REQ, Q_RETRIEVE_VIS_REQ,
                                         retrieve_request, push, pop)
           └── redis              (runtime — Part 2 Docker service)

External services (Part 2 Docker stack, accessed via Redis queues):
  text_indexer   — sentence-transformers + Qdrant
  visual_indexer — ColQwen2 / Byaldi
  ollama         — Gemma 3 4B (LLM inference)
```

---

## Key design choices

| Choice | Rationale |
|--------|-----------|
| `AsyncOpenAI` instead of `OpenAI` | Enables `await` on LLM calls without blocking the event loop; required for `async for chunk in stream` |
| `asyncio.to_thread` for `rag_query` | `rag_query` uses blocking Redis `brpop` and spawns threads internally — it cannot be made async without rewriting Part 3; offloading to a thread is the minimal, non-invasive solution |
| `StreamingResponse` (not `EventSource` on server) | FastAPI's built-in `StreamingResponse` with an async generator is sufficient; avoids the `sse-starlette` dependency |
| Single `ui.html` file | No build toolchain; loaded once at startup and cached in `_UI`; simplest possible delivery |
| No modifications to Part 2 / Part 3 | Part 4 only imports from Part 3; all Part 2 interaction goes through the existing Redis queue protocol |
