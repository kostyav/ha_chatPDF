# Part 2 — Multimodal RAG System

## Task

Build a lightweight RAG system that answers natural-language questions from the
[VisualMRC dataset](https://arxiv.org/pdf/2304.06447) using PDFs that contain
text, tables, and technical drawings (chemical synthesis schemes, epidemiological
maps).  Optimised for a T4 GPU (16 GB VRAM).

---

## Core Stack

| Role | Technology |
|------|-----------|
| Parser | Docling ≥ 2.28 — layout-aware extraction; exports tables as Markdown, pages as images |
| Text retriever | sentence-transformers `all-MiniLM-L6-v2` + Qdrant vector store |
| Visual retriever | ColQwen2-2B via Byaldi — indexes page-image patches |
| Generator | Gemma 3 4B (`q4_K_M` GGUF) via Ollama / llama.cpp / vLLM |
| Message queue | Redis 7 (LPUSH / BRPOP FIFO) |

The inference engine is **configurable at deploy time** — all three backends
expose an OpenAI-compatible `/v1/chat/completions` endpoint, so no code changes
are needed to switch between them.

---

## Dataset

| Item | Location |
|------|----------|
| 10 source PDFs | `src/part2/example_data/*.pdf` |
| Ground-truth Q&A | `src/part2/example_data/train_dataframe_subset.csv` |

Source: [Kaggle PdfVQA competition](https://www.kaggle.com/competitions/pdfvqa/overview),
ArXiv 2304.06447.

---

## Deliverables

### 1. Dataset
10 PDFs from the VisualMRC dataset in `example_data/`.  Ground truth in
`train_dataframe_subset.csv` (columns: `question`, `answer`, `pmcid`).

### 2. Complete RAG code
Microservices architecture in `services/` — see [ARCHITECTURE.md](ARCHITECTURE.md).
Single-process fallback in `rag/` for local development and unit tests.

### 3. Retrieval log

Every query issued through `query.py` automatically appends one JSON line to
`/data/logs/retrieval_log.jsonl` (persisted in the `query-logs` Docker volume):

```json
{
  "timestamp": "2026-03-31T11:11:38Z",
  "query": "Which section describes Fig. 4?",
  "best_score": 0.240,
  "chunks": [
    {"score": 0.240, "pdf_id": "23870758", "text": "…full chunk text…"}
  ],
  "images": ["/data/logs/images/f7c44538_doc0_page3.png"]
}
```

For queries that score **below the similarity threshold** the entry is still
written with `chunks: []` and `images: []`.

A **PDF report** can be generated from the log at any time — see [Generating the
PDF report](#generating-the-pdf-report) below.

A **pytest suite** for the log is in `tests/part2/test_retrieval_log.py`:

```bash
pytest tests/part2/test_retrieval_log.py               # unit tests (no server)
pytest tests/part2/test_retrieval_log.py -m integration # writes retrieval_log.json
```

---

## How to Run

### 1. Configure the LLM engine

Edit `src/part2/config/config.yaml`:

```yaml
engine: ollama          # ollama | llamacpp | vllm
model: gemma3:4b
quantization: q4_K_M
embedding_model: sentence-transformers/all-MiniLM-L6-v2
```

Per-engine example configs are in `src/part2/config/`.

### 2. Pull the Ollama model

```bash
docker compose -f src/part2/docker-compose.yml up -d ollama
docker compose -f src/part2/docker-compose.yml exec ollama ollama pull gemma3:4b
```

### 3. Copy PDFs into the shared volume

```bash
docker volume create rag-part2_pdf-data
docker run --rm \
  -v rag-part2_pdf-data:/data/pdfs \
  -v $(pwd)/src/part2/example_data:/src:ro \
  alpine sh -c "ls /src/*.pdf | head -2 | xargs -I{} cp -v {} /data/pdfs/"
```

> Start with 2 PDFs on a weak machine — indexing all 10 takes significant time.

### 4. Start all services (indexes PDFs, then stays ready)

```bash
cd src/part2
docker compose up --build
```

The orchestrator indexes all PDFs and exits.  The other services (redis, qdrant,
text\_indexer, visual\_indexer, ollama) remain running.

### 5. Run queries

```bash
# Interactive REPL
docker compose run --rm -it orchestrator python query.py

# Single question
docker compose run --rm orchestrator \
  python query.py --question "Which section describes Fig. 4?"
```

Each query prints the answer, best score, retrieved text chunks, and the paths of
any saved page images.  All results are also appended to `retrieval_log.jsonl`.

### 6. Generating the PDF report

```bash
docker compose run --rm orchestrator python report.py
```

This reads `/data/logs/retrieval_log.jsonl` and writes `/data/logs/report.pdf`
— one page per query, showing the question, retrieved chunks, and inline page
images.

Copy outputs to the host:

```bash
docker run --rm \
  -v rag-part2_query-logs:/data/logs \
  -v $(pwd):/out \
  alpine sh -c "cp /data/logs/report.pdf /out/ && cp -r /data/logs/images /out/"
```

### 7. Run tests

```bash
pytest tests/part2 -m "not integration"   # unit tests — no server needed
pytest tests/part2 -m integration         # requires live Ollama server
```

---

## Switching the LLM backend

Override `LLM_BASE_URL` and `LLM_MODEL` at runtime:

```bash
LLM_BASE_URL=http://vllm:8000/v1 LLM_MODEL=google/gemma-3-4b-it \
  docker compose up orchestrator
```

---

## Single-process mode (no Docker)

The `rag/` module runs everything in one process — useful for development and
unit tests without the full service stack.  Requires Ollama running locally.

```bash
pip install -r src/part2/requirements.txt && pip install -e .
ollama serve && ollama pull gemma3:4b

python -m src.part2.rag.pipeline \
  --pdf-dir src/part2/example_data \
  --question "Which section describes Fig. 4?"
```

---

## Evaluation

```bash
python src/part2/evaluate.py \
  --csv  src/part2/example_data/train_dataframe_subset.csv \
  --pdf-dir src/part2/example_data \
  --output eval_results.json
```

Iterates every row in the CSV, runs the RAG pipeline, and scores answers with
BERTScore F1 against ground truth.  Results are written to `eval_results.json`.

---

## GPU Memory Budget (T4 16 GB)

| Service | When active | VRAM |
|---------|------------|------|
| parser (Docling) | indexing only | ~4–5 GB |
| visual\_indexer (ColQwen2) | indexing + query | ~3–4 GB |
| ollama (Gemma 3 4B) | query only | ~2.5 GB |
| text\_indexer + Qdrant | always | CPU only |

Parser and Ollama are naturally time-separated, so peak GPU usage is ~4–5 GB.
