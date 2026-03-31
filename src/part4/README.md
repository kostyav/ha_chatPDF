# Part 4: API Serving & Streaming

## Task
Wrap the agentic flow in a production-ready web interface.

## Framework
Python FastAPI.

## Endpoint
`POST /chat` — accepts `{"question": "..."}` and streams the agent’s response via Server-Sent Events.

## Response Streaming
Real-time streaming via Server-Sent Events (SSE). The user sees the agent’s response as it is generated — not after the full block of text is ready.

## Deliverable
The application code.

## Constraints and rules for code generation
1. Use the code from `src/part3` to understand the flow of the agent.
2. Use the code from `src/part2` to understand the inference process.
3. All the code (LOC) must be kept as minimal as possible. Use 3rd party libraries as much as possible to minimize the LOC.

The code must be kept in `src/part4`. Add more subfolders to keep the layout clean and simple.

---

## Implementation

### Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application — routes, SSE generator, request model |
| `ui.html` | Single-page chat UI served at `GET /` |
| `Dockerfile` | Container image (python:3.12-slim + fastapi + uvicorn) |
| `docker-compose.yml` | Service definition; joins `rag-part2_rag-net` |

### How to run

```bash
# Part 2 Docker stack must be running first (redis, ollama, text_indexer, visual_indexer)
cd src/part4
docker compose up --build
```

Open `http://localhost:8080` in a browser.

Or query the API directly:

```bash
curl -N -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d ‘{"question": "What does Fig. 4 show?"}’
```

### SSE event types

| Event | Payload | Meaning |
|-------|---------|---------|
| `status` | `"string"` | Progress update (routing / retrieving / synthesizing) |
| `token` | `"string"` | One token of the LLM’s streamed output |
| `done` | `""` | Stream finished |

### Environment variables

Inherited from `src/part3/agent.py` — all configurable via Docker Compose `environment:`.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama / vLLM / llama.cpp endpoint |
| `LLM_MODEL` | `gemma3:4b` | Model name |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum retrieval score to use RAG |
| `TOP_K` | `3` | Number of chunks/images to retrieve |
| `RETRIEVE_TIMEOUT` | `60` | Seconds to wait for retrieval services |
