# CLAUDE.md — Project Knowledge Base

## What This Project Is

A two-part AI assessment demonstrating local LLM deployment and multimodal RAG:

- **Part 1:** Deploy a local LLM across three pluggable inference backends (Ollama, llama.cpp, vLLM), all exposed via OpenAI-compatible API.
- **Part 2:** A multimodal RAG system that ingests scientific PDFs and answers questions using both text retrieval (Qdrant) and visual retrieval (ColQwen2), coordinated via Redis message queues in Docker microservices.

---

## Repository Structure

```
/
├── CLAUDE.md                          # This file
├── Makefile                           # Docker image caching for Lightning.ai
├── pyproject.toml                     # Python package (makes src.* importable)
├── requirements.txt                   # Top-level Python deps
├── general_task_description.md        # Assessment spec
│
├── src/
│   ├── part1/                         # LLM deployment abstraction
│   │   ├── engines/factory.py         # Factory: engine → OpenAI client + base URL
│   │   ├── config/config.yaml         # Active config (engine, model, quantization)
│   │   ├── deploy.sh                  # Launches engine natively or via Docker
│   │   └── docker-compose.yml         # Profiles: ollama | llamacpp | vllm
│   │
│   └── part2/                         # Multimodal RAG system
│       ├── docker-compose.yml         # 7 services: parser, text_indexer,
│       │                              #   visual_indexer, orchestrator,
│       │                              #   redis, qdrant, ollama
│       ├── shared/schemas.py          # Redis queue names + message constructors
│       ├── rag/                       # Single-process RAG (for tests)
│       │   ├── parser.py              # Docling PDF → markdown + images
│       │   ├── indexer.py             # FAISS + Byaldi dual-index
│       │   └── pipeline.py            # End-to-end query pipeline
│       └── services/                  # Docker microservices
│           ├── parser/main.py         # Consumes parse.requests → emits parse.results
│           ├── text_indexer/main.py   # sentence-transformers + Qdrant upserts
│           ├── visual_indexer/main.py # ColQwen2-2B via Byaldi, GPU
│           ├── orchestrator/main.py   # Indexing pipeline coordinator (service CMD)
│           └── orchestrator/query.py  # Standalone query client (run with -it)
│
└── tests/
    ├── part1/                         # Unit + integration tests for Part 1
    └── part2/                         # Unit + integration tests for Part 2
```

---

## Technology Stack

### Part 1 — LLM Inference
| Engine | Port | Format |
|--------|------|--------|
| Ollama | 11434 | Automatic quantization |
| llama.cpp | 8080 | GGUF |
| vLLM | 8000 | HuggingFace safetensors |

All three expose OpenAI-compatible `/v1/chat/completions`. The `factory.py` maps engine names to base URLs — no engine-specific code anywhere else.

### Part 2 — RAG Stack
| Component | Technology | Notes |
|-----------|-----------|-------|
| PDF Parser | Docling >=2.28 | Layout-aware, exports tables as Markdown |
| Text Embeddings | all-MiniLM-L6-v2 | 384-dim, CPU |
| Vector DB | Qdrant | HNSW index, persistent volume |
| Visual Embeddings | ColQwen2-2B (Byaldi) | Page-image patches, GPU |
| LLM Generator | Gemma 3 4B q4_K_M | Via Ollama, ~2.5 GB VRAM |
| Message Queue | Redis 7 (Alpine) | LPUSH/BRPOP FIFO queues |

### Why Microservices?
Docling requires `transformers>=4.48` and Byaldi requires `transformers>=4.47,<4.48` — they cannot share a Python environment. Each runs in its own container.

---

## How to Run

### Part 1

```bash
pip install -r requirements.txt && pip install -e .
# Configure: edit src/part1/config/config.yaml  (engine: ollama|llamacpp|vllm)
./src/part1/deploy.sh                              # native
./src/part1/deploy.sh src/part1/config/config.ollama.yaml --docker   # Docker
```

Tests:
```bash
pytest tests/part1 -m "not integration"   # no server needed
pytest tests/part1 -m integration          # requires running server
```

### Part 2

```bash
cd src/part2
docker compose up --build
# In another terminal, query:
docker compose run --rm -it orchestrator python query.py
docker compose run --rm orchestrator python query.py --question "What does Fig. 4 show?"
```

Docker image caching (Lightning.ai — cache lost on restart):
```bash
make build   # build & save images to .docker_images/
make up      # load cached images, start services
make down    # stop
```

Tests:
```bash
pytest tests/part2 -m "not integration"   # unit tests only
pytest tests/part2 -m integration          # full stack required
```

---

## Redis Queue Architecture (Part 2)

```
orchestrator
  │  LPUSH parse.requests {pdf_path, pdf_id}
  ↓
parser ──BRPOP parse.requests──► docling parse ──LPUSH parse.results──►
                                                                        text_indexer ──► Qdrant
visual_indexer ──BRPOP parse.results──► Byaldi index (GPU)

# Query flow (per-query correlation ID = UUID):
orchestrator LPUSH retrieve.text.requests  {query, corr_id}
             LPUSH retrieve.visual.requests {query, corr_id}
text_indexer  → LPUSH res.text.<corr_id>
visual_indexer → LPUSH res.visual.<corr_id>
orchestrator BRPOP both → combine context → Ollama → answer
```

---

## Key Design Decisions

1. **Similarity threshold gating (0.3):** If best retrieval score < 0.3, returns "The document does not contain information about this query." Prevents hallucination.
2. **Idempotent indexing:** Chunk IDs = `UUID(md5(pdf_id + text))` — safe to re-index; duplicates are upserted not duplicated.
3. **Chunking strategy:** Split by `## headers` first; if section > 800 chars, split by paragraph.
4. **GPU budget (T4 16 GB):** Parser (4–5 GB, indexing only) + ColQwen2 (3–4 GB) + Gemma (2.5 GB, generation only) — naturally time-separated, peak ~5 GB.
5. **`rag/` single-process module** kept for unit tests without Docker.

---

## Model Used

**Gemma 3 4B** (GGUF q4_K_M, ~2.5 GB VRAM) via Ollama. Configured in `src/part2/config/config.yaml`.

HuggingFace login required for gated model access:
```bash
huggingface-cli login --token <token>
```

---

## Evaluation

```bash
python src/part2/evaluate.py
```

Uses BERTScore (F1) against ground truth from `src/part2/pdfvqa_prep_work/train_dataframe_subset.csv` (10 PDFs from VisualMRC dataset, ArXiv 2304.06447).

---

## Test Markers

- `@pytest.mark.integration` — requires live services (server/Docker)
- Tests without this marker run fully in isolation (mocks, fixtures)

Run only unit tests: `pytest -m "not integration"`
