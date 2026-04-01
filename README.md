# ha_chatPDF

A multi-part AI engineering project demonstrating local LLM deployment, multimodal RAG, agentic orchestration, a streaming web API, quantization benchmarking, and Kubernetes deployment.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Part 1 — Local LLM Deployment](#part-1--local-llm-deployment)
- [Part 2 — Multimodal RAG System](#part-2--multimodal-rag-system)
- [Part 3 — Agentic Orchestrator](#part-3--agentic-orchestrator)
- [Part 4 — Streaming Web API](#part-4--streaming-web-api)
- [Part 5.2 — Quantization & Performance Profiling](#part-52--quantization--performance-profiling)
- [Part 5.3 — Kubernetes Deployment](#part-53--kubernetes-deployment)
- [Lightning.ai — Preventing Container Rebuilds](#lightningai--preventing-container-rebuilds)

---

## Project Overview

| Part | Description |
|------|-------------|
| **1** | Deploy a local LLM across three pluggable inference backends (Ollama, llama.cpp, vLLM), all exposed via an OpenAI-compatible REST API |
| **2** | Multimodal RAG system that ingests scientific PDFs and answers questions using text retrieval (Qdrant + sentence-transformers) and visual retrieval (ColQwen2), coordinated via Redis message queues in Docker microservices |
| **3** | Autonomous agent that wraps the Part 2 RAG system as a callable tool with two-step prompt-based routing — no LangChain/LangGraph required |
| **4** | FastAPI layer that streams the agent's responses token-by-token to a browser via Server-Sent Events (SSE) |
| **5.2** | Quantization benchmark: Gemma-3 model family across bit-widths on an NVIDIA T4, comparing TPS, VRAM, and output quality |
| **5.3** | Kubernetes manifests (Kustomize) that port the full Docker Compose stack to Minikube, mapping every service 1-to-1 |

---

## Repository Structure

```
/
├── CLAUDE.md                          # Project knowledge base
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
│   │   ├── docker-compose.yml         # Profiles: ollama | llamacpp | vllm
│   │   ├── README.md
│   │   └── ARCHITECTURE.md
│   │
│   ├── part2/                         # Multimodal RAG system
│   │   ├── docker-compose.yml         # 7 services: parser, text_indexer,
│   │   │                              #   visual_indexer, orchestrator,
│   │   │                              #   redis, qdrant, ollama
│   │   ├── shared/schemas.py          # Redis queue names + message constructors
│   │   ├── rag/                       # Single-process RAG (for tests/dev)
│   │   │   ├── parser.py              # Docling PDF → markdown + images
│   │   │   ├── indexer.py             # FAISS + Byaldi dual-index
│   │   │   └── pipeline.py            # End-to-end query pipeline
│   │   ├── services/                  # Docker microservices
│   │   │   ├── parser/main.py
│   │   │   ├── text_indexer/main.py
│   │   │   ├── visual_indexer/main.py
│   │   │   ├── orchestrator/main.py   # Indexing pipeline coordinator
│   │   │   └── orchestrator/query.py  # Standalone query client
│   │   ├── evaluate.py                # BERTScore evaluation
│   │   ├── example_data/              # 10 source PDFs (VisualMRC subset)
│   │   ├── README.md
│   │   └── ARCHITECTURE.md
│   │
│   ├── part3/                         # Agentic orchestrator
│   │   ├── agent.py                   # Agent loop + RAG tool + trace (~190 LOC)
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   ├── README.md
│   │   └── ARCHITECTURE.md
│   │
│   ├── part4/                         # Streaming web API
│   │   ├── main.py                    # FastAPI server (~80 LOC)
│   │   ├── ui.html                    # Single-file chat UI (~80 LOC)
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   ├── README.md
│   │   └── ARCHITECTURE.md
│   │
│   ├── part5_2/                       # Quantization benchmark
│   │   ├── benchmark.py
│   │   ├── run_questions.py
│   │   ├── results.json
│   │   ├── bert_results.json
│   │   ├── REPORT.md                  # Full performance report
│   │   └── README.md
│   │
│   └── part5_3/                       # Kubernetes deployment
│       ├── k8s/                       # Kustomize manifests
│       │   ├── kustomization.yaml
│       │   ├── namespace.yaml
│       │   ├── configmap.yaml
│       │   ├── secret.yaml
│       │   ├── pvc/                   # PersistentVolumeClaims
│       │   ├── redis/
│       │   ├── qdrant/
│       │   ├── ollama/
│       │   ├── parser/
│       │   ├── text-indexer/
│       │   ├── visual-indexer/
│       │   ├── orchestrator/
│       │   │   ├── indexer-job.yaml
│       │   │   └── query-job.yaml
│       │   └── api/                   # FastAPI NodePort service
│       ├── config/config.yaml
│       └── README.md
│
└── tests/
    ├── part1/                         # Unit + integration tests
    └── part2/                         # Unit + integration tests
```

---

## Part 1 — Local LLM Deployment

Deploys a local LLM with a thin abstraction over three inference engines. All three expose an OpenAI-compatible `/v1/chat/completions` endpoint; the rest of the project treats them identically.

### Technology

| Engine | Port | Model format |
|--------|------|--------------|
| Ollama | 11434 | Automatic quantization |
| llama.cpp | 8080 | GGUF |
| vLLM | 8000 | HuggingFace safetensors |

The `engines/factory.py` (~20 LOC) maps an engine name to a base URL and returns a pre-configured `openai.OpenAI` client — no engine-specific code anywhere else.

### Install

```bash
pip install -r requirements.txt
pip install -e .
```

> **HuggingFace authentication:** `google/gemma-3-1b-it` and the llama.cpp GGUF counterpart are gated models. Accept their terms and authenticate before downloading:
> ```bash
> huggingface-cli login --token <your_token>
> ```

### Configure

Edit `src/part1/config/config.yaml`:

```yaml
engine: ollama          # ollama | llamacpp | vllm
model: gemma:3-1b-it
quantization: q4_0
```

Per-engine example configs are in `src/part1/config/`.

### Start the inference server

**Native (engine must be installed on host):**

```bash
./src/part1/deploy.sh                                         # uses config.yaml
./src/part1/deploy.sh src/part1/config/config.ollama.yaml     # explicit config
./src/part1/deploy.sh src/part1/config/config.llamacpp.yaml
./src/part1/deploy.sh src/part1/config/config.vllm.yaml
```

**Docker:**

```bash
./src/part1/deploy.sh src/part1/config/config.ollama.yaml   --docker
./src/part1/deploy.sh src/part1/config/config.llamacpp.yaml --docker
./src/part1/deploy.sh src/part1/config/config.vllm.yaml     --docker
```

### Tests

```bash
pytest tests/part1 -m "not integration"   # unit tests — no server needed
pytest tests/part1 -m integration          # requires running server
```

---

## Part 2 — Multimodal RAG System

A document question-answering system over scientific PDFs combining text retrieval (dense vectors in Qdrant) and visual retrieval (page-image patches via ColQwen2).

### Why microservices?

Docling requires `transformers>=4.48` and Byaldi requires `transformers>=4.47,<4.48` — they cannot share a Python environment. Each runs in its own container, communicating over Redis FIFO queues.

### Technology stack

| Component | Technology |
|-----------|-----------|
| PDF Parser | Docling ≥2.28 — layout-aware, exports tables as Markdown |
| Text embeddings | `all-MiniLM-L6-v2` (384-dim, CPU) |
| Vector DB | Qdrant — HNSW index, persistent volume |
| Visual embeddings | ColQwen2-2B via Byaldi — page-image patches, GPU |
| LLM generator | Gemma 3 4B q4_K_M via Ollama (~2.5 GB VRAM) |
| Message queue | Redis 7 Alpine — LPUSH/BRPOP FIFO queues |

### Redis queue architecture

```
orchestrator
  │  LPUSH parse.requests {pdf_path, pdf_id}
  ▼
parser ──BRPOP──► docling parse ──LPUSH parse.results──►
                                                        text_indexer ──► Qdrant
visual_indexer ──BRPOP parse.results──► Byaldi index (GPU)

# Query flow (per-query correlation ID = UUID):
orchestrator  LPUSH retrieve.text.requests   {query, corr_id}
              LPUSH retrieve.visual.requests  {query, corr_id}
text_indexer  → LPUSH res.text.<corr_id>
visual_indexer → LPUSH res.visual.<corr_id>
orchestrator  BRPOP both → combine context → Ollama → answer
```

### Key design decisions

- **Similarity threshold (0.3):** If the best retrieval score is below 0.3, returns "The document does not contain information about this query." Prevents hallucination on out-of-scope questions.
- **Idempotent indexing:** Chunk IDs = `UUID(md5(pdf_id + text))` — safe to re-index; duplicates are upserted.
- **Chunking:** Split by `## headers` first; if a section exceeds 800 chars, split by paragraph.
- **GPU budget (T4 16 GB):** Parser (4–5 GB, indexing only) + ColQwen2 (3–4 GB) + Gemma (2.5 GB, generation only) — naturally time-separated, peak ~5 GB.

### Install

```bash
pip install -r src/part2/requirements.txt
pip install -e .
```

### Pull model and copy PDFs

```bash
docker compose -f src/part2/docker-compose.yml up -d ollama
docker compose -f src/part2/docker-compose.yml exec ollama ollama pull gemma3:4b

docker volume create rag-part2_pdf-data
docker run --rm \
  -v rag-part2_pdf-data:/data/pdfs \
  -v $(pwd)/src/part2/example_data:/src:ro \
  alpine sh -c "ls /src/*.pdf | head -2 | xargs -I{} cp -v {} /data/pdfs/"
```

### Start all services

```bash
docker compose -f src/part2/docker-compose.yml up --build
```

### Query

```bash
# Interactive REPL
docker compose -f src/part2/docker-compose.yml run --rm -it orchestrator python query.py

# Single question
docker compose -f src/part2/docker-compose.yml run --rm orchestrator \
  python query.py --question "Which section describes Fig. 4?"
```

### Evaluate

```bash
python src/part2/evaluate.py
```

Uses BERTScore (F1) against ground truth from `src/part2/pdfvqa_prep_work/train_dataframe_subset.csv` (10 PDFs from the VisualMRC dataset, ArXiv 2304.06447).

### Tests

```bash
pytest tests/part2 -m "not integration"   # unit tests — no server needed
pytest tests/part2 -m integration          # requires live services
```

---

## Part 3 — Agentic Orchestrator

An autonomous agent that wraps the Part 2 RAG pipeline as a callable tool. The agent routes each user question through a two-step prompt loop — no LangChain or LangGraph is needed because `gemma3:4b` does not support the OpenAI `tools` API parameter.

### Agent decision flow

```
User question
      │
      ▼
┌──────────────────────────────────────────┐
│  Step 1 — Router (LLM call, max_tokens=5) │
│  "Does this require searching scientific  │
│   PDF documents? Reply YES or NO."        │
└──────────────┬───────────────────────────┘
               │
       ┌───────┴────────┐
       ▼ YES             ▼ NO
  rag_query()       Direct LLM answer
       │            (general knowledge)
       ▼
  Step 2a — Synthesis LLM call
  (RAG context + original question)
```

### LLM calls per query

| Step | Purpose | max_tokens |
|------|---------|-----------|
| Router | YES/NO routing decision | 5 |
| RAG synthesis (if YES) | Grounded answer from retrieved context | 512 |
| Direct answer (if NO) | Answer from general knowledge | 512 |

### Run

Start Part 2 stack first, then:

```bash
docker compose -f src/part3/docker-compose.yml run --rm -it agent python agent.py
docker compose -f src/part3/docker-compose.yml run --rm agent \
  python agent.py --question "What does Fig. 4 show?"
```

The agent prints `[trace]` lines at every decision point (router decision, tool invocation, Redis hits, scores, LLM calls).

---

## Part 4 — Streaming Web API

A thin FastAPI layer over Part 3 that streams token-by-token responses to the browser via Server-Sent Events (SSE).

### Architecture

```
Browser
  │  GET /         → ui.html (single-file chat UI, no build step)
  │  POST /chat    → StreamingResponse (text/event-stream)
  ▼
main.py (_stream async generator)
  ├── AsyncOpenAI  — router call (step 1, max_tokens=5)
  ├── asyncio.to_thread(agent.rag_query)  — Redis + LLM (blocking, offloaded)
  └── AsyncOpenAI(stream=True)  — final answer, token-by-token
```

SSE event types:

| Event | Payload | Browser action |
|-------|---------|---------------|
| `status` | Italic label text | Replace status label above bubble |
| `token` | One token string | Append to agent bubble |
| `done` | `""` | Remove status label |

### Run

Start Part 2 stack first, then:

```bash
docker compose -f src/part4/docker-compose.yml up --build
# UI available at http://localhost:8080
```

Stream a response via curl:

```bash
curl -N -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Fig. 4 show?"}'
```

---

## Part 5.2 — Quantization & Performance Profiling

Benchmarks Gemma-3 variants on an NVIDIA T4 (15 360 MiB VRAM) via Ollama, measuring tokens per second and peak VRAM/RAM usage.

### Results

| Model | Params | Quantization | Avg TPS | Peak VRAM (MiB) | Peak RAM (MiB) |
|-------|--------|--------------|---------|-----------------|----------------|
| gemma3:270m | 268 M | Q8_0 | **216.1** | 5 287 | 30 |
| gemma3:1b | 1 B | Q4_K_M | 135.9 | 6 297 | 30 |
| gemma3:1b-it-qat | 1 B (QAT) | Q4_0 | 114.7 | **7 493** | 30 |

### Key findings

- **Parameter count dominates speed** more than bit-width. Halving parameters (1 B → 270 M) gains +59 % TPS even when moving from 4-bit to 8-bit precision.
- **Q4_K_M outperforms Q4_0** at the same parameter count — the mixed-precision K-quant format is better aligned with T4 tensor cores.
- **VRAM** scales with `params × bits_per_weight` plus KV-cache footprint; the QAT model uses more VRAM than expected due to longer generated sequences.

### Recommendations

| Goal | Recommended |
|------|------------|
| Maximum throughput / real-time UX | `gemma3:270m` Q8_0 — 216 TPS |
| Best quality per VRAM dollar | `gemma3:1b` Q4_K_M — 136 TPS, richest output |
| Lowest VRAM footprint | `gemma3:270m` Q8_0 — 5 287 MiB |

Full report: [src/part5_2/REPORT.md](src/part5_2/REPORT.md)

---

## Part 5.3 — Kubernetes Deployment

Ports the full Docker Compose stack (Parts 2–4) to a local Kubernetes cluster on Minikube. All service names are identical to Compose so no application code changes were needed.

### Service mapping

| Docker Compose service | K8s workload | K8s Service | Port |
|------------------------|-------------|-------------|------|
| redis | Deployment | ClusterIP | 6379 |
| qdrant | Deployment | ClusterIP | 6333 |
| ollama | Deployment | ClusterIP | 11434 |
| parser | Deployment | — | — |
| text_indexer | Deployment | — | — |
| visual_indexer | Deployment | — | — |
| orchestrator (main.py) | **Job** | — | — |
| orchestrator (query.py) | **Job** | — | — |
| api (Part 4) | Deployment | **NodePort** | 30808 |

Parser, text_indexer, and visual_indexer communicate only via Redis queues — no K8s Service is needed for them.

### Prerequisites

```bash
# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

### Deploy

**Step 1 — Start Minikube**

```bash
minikube start --memory=16g --cpus=6 --disk-size=40g
# With GPU:
minikube start --memory=16g --cpus=6 --disk-size=40g --driver=docker --gpus all
minikube addons enable nvidia-gpu-device-plugin
```

**Step 2 — Build images into Minikube**

```bash
eval $(minikube docker-env)

docker build -t rag/parser:latest         -f src/part2/services/parser/Dockerfile        src/part2/
docker build -t rag/text-indexer:latest   -f src/part2/services/text_indexer/Dockerfile  src/part2/
docker build -t rag/visual-indexer:latest -f src/part2/services/visual_indexer/Dockerfile src/part2/
docker build -t rag/orchestrator:latest   -f src/part2/services/orchestrator/Dockerfile  src/part2/
docker build -t rag/api:latest            -f src/part4/Dockerfile                         .
```

**Step 3 — HuggingFace token (gated models)**

```bash
kubectl create secret generic hf-token \
  --from-literal=HUGGING_FACE_HUB_TOKEN=<YOUR_HF_TOKEN> \
  -n rag-system
```

**Step 4 — Copy PDFs into the cluster**

```bash
kubectl run pdf-loader --image=busybox --restart=Never \
  --overrides='{"spec":{"volumes":[{"name":"pdf","persistentVolumeClaim":{"claimName":"pdf-data"}}],"containers":[{"name":"pdf-loader","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"pdf","mountPath":"/data/pdfs"}]}]}}' \
  -n rag-system

kubectl cp src/part2/example_data/23870758.pdf rag-system/pdf-loader:/data/pdfs/
kubectl cp src/part2/example_data/24069913.pdf rag-system/pdf-loader:/data/pdfs/
kubectl delete pod pdf-loader -n rag-system
```

**Step 5 — Deploy**

```bash
kubectl apply -k src/part5_3/k8s/

kubectl rollout status deployment/redis          -n rag-system
kubectl rollout status deployment/qdrant         -n rag-system
kubectl rollout status deployment/ollama         -n rag-system
kubectl rollout status deployment/parser         -n rag-system
kubectl rollout status deployment/text-indexer   -n rag-system
kubectl rollout status deployment/visual-indexer -n rag-system
```

**Step 6 — Index documents**

```bash
kubectl apply -f src/part5_3/k8s/orchestrator/indexer-job.yaml -n rag-system
kubectl logs -f job/orchestrator-indexer -n rag-system
```

**Step 7 — Query**

```bash
# One-shot (edit args in query-job.yaml first)
kubectl apply -f src/part5_3/k8s/orchestrator/query-job.yaml -n rag-system
kubectl logs -f job/orchestrator-query -n rag-system
kubectl delete job orchestrator-query -n rag-system
```

**Step 8 — Access the API**

```bash
minikube service api -n rag-system --url
# or
curl http://$(minikube ip):30808/

curl -N -X POST http://$(minikube ip):30808/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Fig. 4 show?"}'
```

**Teardown**

```bash
kubectl delete namespace rag-system
minikube stop
```

---

## Lightning.ai — Preventing Container Rebuilds

On Lightning.ai the Docker layer cache lives in ephemeral instance storage and is wiped on every machine restart. Build once and save images to the persistent workspace filesystem.

### First time (or after code changes)

```bash
make build   # build all service images and save to .docker_images/
make pull    # save redis, qdrant, and ollama images too
```

> Add `.docker_images/` to `.gitignore` — the tar files are large:
> ```bash
> echo ".docker_images/" >> .gitignore
> ```

### Every subsequent restart

```bash
make up      # load images from .docker_images/*.tar, then start all services
```

`make up` automatically falls back to a full build if no cached images are found.

### Make targets

| Target | Description |
|--------|-------------|
| `make build` | Build service images and save to `.docker_images/` |
| `make pull` | Pull and save infrastructure images (Redis, Qdrant, Ollama) |
| `make load` | Load all saved `.tar` images into Docker (without starting) |
| `make up` | Load cached images (or build) then start all services |
| `make down` | Stop and remove containers |
