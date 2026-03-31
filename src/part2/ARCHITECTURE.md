# Architecture — Part 2: Multimodal RAG System

## Overview

The system answers natural-language questions about scientific PDFs by combining
two complementary retrieval strategies with a multimodal LLM generator:

- **Text retrieval** — sentence-transformers embeddings stored in a Qdrant vector
  store capture explicit content (tables, methods, results).
- **Visual retrieval** — ColQwen2-2B via Byaldi indexes page images as visual
  patches, capturing content invisible to text embeddings (figures, chemical
  schemes, epidemiological maps).
- **Generation** — Gemma 3 4B receives the retrieved text chunks and up to two
  page images as a multimodal prompt and produces a grounded answer.

All three LLM inference backends (Ollama, llama.cpp, vLLM) expose an
OpenAI-compatible REST API, so the generator is backend-agnostic.

---

## Why Docker + Redis?

The two retrieval libraries have **mutually incompatible `transformers` pins**:

| Service         | Library              | `transformers` constraint        |
|-----------------|----------------------|----------------------------------|
| `parser`        | Docling              | `>=4.48` (needs `rt_detr_v2`)    |
| `visual_indexer`| Byaldi / colpali-engine | `>=4.47, <4.48`               |
| `text_indexer`  | sentence-transformers | no conflict                     |
| `orchestrator`  | openai client only    | no conflict                     |

Each service runs in its own container with its own Python environment, and they
communicate exclusively through **Redis Lists** (LPUSH / BRPOP FIFO queues) —
no shared memory, no conflicting imports.  Redis requires zero configuration
beyond `redis-server --appendonly yes`.

---

## File Layout

```
src/part2/
├── ARCHITECTURE.md          ← this file
├── README.md                ← task specification
├── docker-compose.yml       ← spins up Redis + all four services + Ollama
│
├── shared/                  ← Redis queue names and message constructors
│   ├── __init__.py
│   └── schemas.py           ← imported by every service via PYTHONPATH=/app
│
├── services/                ← one sub-directory per Docker container
│   ├── parser/
│   │   ├── Dockerfile       ← nvidia/cuda base; transformers>=4.48
│   │   ├── requirements.txt
│   │   └── main.py          ← Redis consumer/producer wrapping Docling
│   │
│   ├── text_indexer/
│   │   ├── Dockerfile       ← python:3.12-slim; no GPU required
│   │   ├── requirements.txt
│   │   └── main.py          ← Qdrant index + sentence-transformers
│   │
│   ├── visual_indexer/
│   │   ├── Dockerfile       ← nvidia/cuda base; transformers>=4.47,<4.48
│   │   ├── requirements.txt
│   │   └── main.py          ← Byaldi/ColQwen2 index
│   │
│   └── orchestrator/
│       ├── Dockerfile       ← python:3.12-slim; no GPU required
│       ├── requirements.txt
│       ├── main.py          ← indexing pipeline coordinator (CMD of the service)
│       └── query.py         ← standalone query client (run separately with -it)
│
├── config/                  ← LLM engine configs (used by orchestrator env vars)
│   ├── config.yaml          ← ACTIVE config
│   ├── config.ollama.yaml
│   ├── config.llamacpp.yaml
│   └── config.vllm.yaml
│
├── rag/                     ← original single-process library (kept for tests)
│   ├── __init__.py
│   ├── parser.py
│   ├── indexer.py
│   └── pipeline.py
│
└── evaluate.py              ← BERTScore evaluation loop

tests/part2/
├── conftest.py
├── test_config.py
├── test_parser.py
├── test_pipeline.py
├── test_configurations.py
└── test_evaluate.py
```

---

## Redis Queues

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Queue                     Producer         Consumer(s)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ parse.requests            orchestrator     parser, visual_indexer           │
│ parse.results             parser           text_indexer                     │
│ index.ready               text_indexer,    orchestrator                     │
│                           visual_indexer                                    │
│ retrieve.text.requests    orchestrator     text_indexer                     │
│ res.text.<corr_id>        text_indexer     orchestrator (per-query list)    │
│ retrieve.visual.requests  orchestrator     visual_indexer                   │
│ res.visual.<corr_id>      visual_indexer   orchestrator (per-query list)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

All queues use LPUSH (producer) + BRPOP (consumer) for strict FIFO ordering.
Query result queues are ephemeral per-correlation Redis lists; the orchestrator
deletes them after receiving both results.  Message schemas are defined in
[shared/schemas.py](shared/schemas.py).

---

## End-to-End Data Flow

### Indexing

```
orchestrator
  │  parse.requests {pdf_path, pdf_id}  ×N
  ├──────────────────────────────────────────► parser
  │                                              │ parse.results {pdf_id, markdown, tables_md}
  │                                              └──────────────────────────────► text_indexer
  │                                                                                  │ index.ready {service:"text", pdf_id}
  │                                                                                  └────────────────────────────────────► orchestrator
  │
  └──────────────────────────────────────────► visual_indexer   (consumes parse.requests directly;
                                                 │               Byaldi indexes PDFs, not parsed text)
                                                 │ index.ready {service:"visual", pdf_id}
                                                 └────────────────────────────────────────────────► orchestrator
```

Orchestrator blocks until both `index.ready` sets cover every expected `pdf_id`.

### Query

```
question
  │
  orchestrator
  │  retrieve.text.requests {correlation_id, question, top_k}
  ├──────────────────────────────────────────────────────────► text_indexer
  │                                                              │ retrieve.text.results {correlation_id, hits}
  │                                                              └──────────────────────────────────────────► orchestrator (queue)
  │
  │  retrieve.visual.requests {correlation_id, question, top_k}
  └──────────────────────────────────────────────────────────► visual_indexer
                                                                 │ retrieve.visual.results {correlation_id, hits}
                                                                 └──────────────────────────────────────────► orchestrator (queue)

  orchestrator collects both results, applies similarity threshold,
  then calls Ollama/llama.cpp/vLLM via OpenAI-compatible REST API.
```

---

## Service Details

### `parser`

- **Base image:** `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- **Key dep:** `docling>=2.28`, `transformers>=4.48` (rt_detr_v2 layout model)
- **Consumes:** `parse.requests`
- **Produces:** `parse.results`
- **Volume:** `/data/pdfs` (read-only), `/data/parsed` (page image output)

### `text_indexer`

- **Base image:** `python:3.12-slim`
- **Key deps:** `sentence-transformers>=3.0`, `qdrant-client>=1.9`
- **Consumes:** `parse.results` (indexing), `retrieve.text.requests` (querying)
- **Produces:** `index.ready`, `retrieve.text.results`
- **Threads:** one per consumer loop; `_TextIndex` is shared and thread-safe via Qdrant's own concurrency
- **Persistence:** vectors are stored in the `qdrant` service (backed by the `qdrant-data` volume). On restart, `_is_indexed(pdf_id)` checks Qdrant before re-embedding — already-indexed PDFs are skipped, so `docker compose up` does not re-index.
- **Idempotency:** each chunk's Qdrant point ID is a deterministic UUID derived from `md5(pdf_id + text)`, so upserts of duplicate content are no-ops.

### `visual_indexer`

- **Base image:** `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- **Key deps:** `byaldi>=0.0.6`, `colpali-engine>=0.3.8`, `transformers>=4.47,<4.48`
- **Consumes:** `parse.requests` (indexing — needs raw PDF path, not parsed text),
  `retrieve.visual.requests` (querying)
- **Produces:** `index.ready`, `retrieve.visual.results`
- **Volume:** `/data/pdfs` (read-only), `/data/byaldi_index` (index storage)

### `orchestrator` (`main.py`)

- **Base image:** `python:3.12-slim`
- **Key deps:** `redis>=5.0`, `qdrant-client>=1.9`
- **Role:** indexing-only coordinator — pushes `parse.requests`, waits for all
  `index.ready` signals, then exits.
- **Background thread:** `_consume_index_ready` (BRPOP loop on `index.ready`)

### `query.py` (standalone client)

- **Same image** as orchestrator (copied into the container at build time)
- **Key deps:** `redis>=5.0`, `openai>=1.0`
- **Role:** query-only client — pushes `retrieve.text.requests` /
  `retrieve.visual.requests`, collects results, calls LLM.  Never touches
  indexing queues or flushes any Redis keys.
- **Query correlation:** each query gets a `uuid4` correlation ID; results are
  delivered to ephemeral per-correlation Redis lists (`res.text.<id>` /
  `res.visual.<id>`) and deleted after receipt.
- **Run with:** `docker compose run --rm -it orchestrator python query.py`

---

## Configuration

Services are configured via environment variables in `docker-compose.yml`:

| Variable              | Service       | Default                        |
|-----------------------|---------------|--------------------------------|
| `REDIS_URL`           | all           | `redis://redis:6379/0`         |
| `DPI`                 | parser        | `300`                          |
| `EMBEDDING_MODEL`     | text_indexer  | `all-MiniLM-L6-v2`             |
| `CHUNK_MAX_CHARS`     | text_indexer  | `800`                          |
| `COLQWEN_MODEL`       | visual_indexer| `vidore/colqwen2-v0.1`         |
| `LLM_BASE_URL`        | orchestrator  | `http://ollama:11434/v1`       |
| `LLM_MODEL`           | orchestrator  | `gemma3:4b`                    |
| `SIMILARITY_THRESHOLD`| orchestrator  | `0.3`                          |
| `TOP_K`               | orchestrator  | `3`                            |
| `RETRIEVE_TIMEOUT`    | orchestrator  | `60` (seconds)                 |

---

## GPU Memory Budget (T4 16 GB)

| Service           | When active     | Allocation  | Notes                              |
|-------------------|-----------------|-------------|-------------------------------------|
| `parser`          | indexing only   | ~4–5 GB     | Docling layout + OCR models         |
| `visual_indexer`  | indexing + query| ~3–4 GB     | ColQwen2-2B; add 4-bit quant to halve|
| `ollama` (Gemma)  | query only      | ~2.5 GB     | `q4_K_M` quantisation               |
| `text_indexer`    | always          | CPU only    | Qdrant runs on CPU                  |
| `orchestrator`    | always          | CPU only    | No ML models                        |

Parser and Ollama are naturally time-separated (indexing vs generation), so peak
GPU usage stays around 4–5 GB on a single T4.

---

## How to Run

### Prerequisites

Per-engine configs are in `src/part2/config/`.

### 3. Pull the Ollama model

```bash
# Start Ollama, then pull the model into the running container
docker compose -f src/part2/docker-compose.yml up -d ollama
docker compose -f src/part2/docker-compose.yml exec ollama ollama pull gemma3:4b

# Copy N or 2  PDFs into the shared volume. Do not move all 10: it takes time on weak machine
docker volume create rag-part2_pdf-data
docker run --rm -v rag-part2_pdf-data:/data/pdfs \
  -v $(pwd)/src/part2/example_data:/src:ro \
  alpine sh -c "ls /src/*.pdf | head -2 | xargs -I{} cp -v {} /data/pdfs/"
```

You can later clean the files with: docker run --rm -v rag-part2_pdf-data:/data/pdfs

### Start all services

```bash
cd src/part2
docker compose up --build
```

### Run a query

`main.py` handles indexing only. Use `query.py` for querying:

```bash
# Interactive REPL (requires -it for stdin)
docker compose run --rm -it orchestrator python query.py

# Single question
docker compose run --rm orchestrator \
  python query.py --question "Which section describes Fig. 4?"
```

### Switch LLM backend

Override `LLM_BASE_URL` and `LLM_MODEL` in `docker-compose.yml` or pass them
as environment overrides:

```bash
LLM_BASE_URL=http://vllm:8000/v1 LLM_MODEL=google/gemma-3-4b-it \
  docker compose up orchestrator
```

---

## Single-process mode (no Docker)

The original `rag/` module remains for unit tests and local development without
Docker.  It has the same chunking logic as `text_indexer` and the same Byaldi
wrapper as `visual_indexer`, but runs everything in one process.

```bash
cd /teamspace/studios/this_studio
python -m src.part2.rag.pipeline \
  --pdf-dir src/part2/example_data \
  --question "Which section describes Fig. 4?"
```

Prerequisite: Ollama must be running locally (`ollama serve && ollama pull gemma3:4b`).
