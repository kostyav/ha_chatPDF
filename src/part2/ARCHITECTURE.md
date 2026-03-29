# Architecture — Part 2: Multimodal RAG System

## Overview

The system answers natural-language questions about scientific PDFs by combining
two complementary retrieval strategies with a multimodal LLM generator:

- **Text retrieval** — sentence-transformers embeddings stored in a FAISS index
  capture explicit content (tables, methods, results).
- **Visual retrieval** — ColQwen2-2B via Byaldi indexes page images as visual
  patches, capturing content that is invisible to text embeddings (figures,
  chemical schemes, epidemiological maps).
- **Generation** — Gemma 3 4B receives the retrieved text chunks and up to two
  page images as a multimodal prompt and produces a grounded answer.

All three LLM inference backends (Ollama, llama.cpp, vLLM) expose an
OpenAI-compatible REST API, so the generator is backend-agnostic — the same
`openai.OpenAI` client is used regardless of the engine.

---

## File Layout

```
src/part2/
├── ARCHITECTURE.md          ← this file
├── README.md                ← task specification
├── requirements.txt         ← all runtime dependencies
│
├── config/                  ← runtime configuration (one file per engine)
│   ├── config.yaml          ← ACTIVE config (edit this to switch engine/model)
│   ├── config.ollama.yaml
│   ├── config.llamacpp.yaml
│   └── config.vllm.yaml
│
├── engines/                 ← engine abstraction (mirrors src/part1/engines/)
│   ├── __init__.py
│   └── factory.py           ← load_config() + get_client() → (OpenAI, model_name)
│
├── rag/                     ← retrieval-augmented generation components
│   ├── __init__.py
│   ├── parser.py            ← Docling: PDF → Markdown + tables + page images
│   ├── indexer.py           ← TextIndex (FAISS) + VisualIndex (Byaldi/ColQwen2)
│   └── pipeline.py          ← RAGPipeline: index_documents() + query()
│
└── evaluate.py              ← BERTScore evaluation loop over ground-truth CSV

tests/part2/
├── conftest.py              ← shared fixtures, server lifecycle
├── test_config.py           ← config + factory unit tests
├── test_parser.py           ← parser unit + integration tests
├── test_pipeline.py         ← pipeline unit + integration tests
├── test_configurations.py   ← per-engine parametrised tests
└── test_evaluate.py         ← evaluation script unit tests
```

---

## Configuration Schema

Every config file is a YAML document with the following keys:

```yaml
engine: ollama                                    # one of: ollama | llamacpp | vllm
model: gemma3:4b                                  # engine-specific identifier
quantization: q4_K_M                              # optional — interpretation varies by engine

embedding_model: sentence-transformers/all-MiniLM-L6-v2  # or BAAI/bge-small-en-v1.5
colqwen_model:   vidore/colqwen2-v0.1             # Byaldi visual retriever

retriever:
  top_k: 3                                        # chunks returned per query
  similarity_threshold: 0.3                       # below this → no-information response
  index_dir: .byaldi_index

parser:
  dpi: 300                                        # page image resolution
  output_dir: .parsed_docs
```

### `model` and `quantization` per engine

| Engine    | `model` value                     | `quantization` effect                        |
|-----------|-----------------------------------|----------------------------------------------|
| ollama    | Ollama tag, e.g. `gemma3:4b`      | Encoded inside the tag; leave empty          |
| llamacpp  | Local `.gguf` path                | Baked into the filename; field is ignored    |
| vllm      | HuggingFace model id              | Passed as `--quantization` CLI flag          |

---

## Component Breakdown

### `engines/factory.py` — engine abstraction

```
get_client(config_path)
│
├── load_config(path)              # reads YAML → dict
│
├── ENGINE_BASE_URLS[engine]       # dict lookup
│   ├── ollama   → :11434/v1
│   ├── llamacpp → :8080/v1
│   └── vllm     → :8000/v1
│
└── OpenAI(base_url=..., api_key="none")
    └── returns (client, model_name)
```

Identical pattern to `src/part1/engines/factory.py`. The rest of the system
only ever calls `get_client()` — the engine choice is fully encapsulated.

---

### `rag/parser.py` — document parsing

Uses **Docling** for layout-aware PDF extraction.

```
parse_pdf(pdf_path, output_dir, dpi)
│
├── DocumentConverter(PdfFormatOption(generate_page_images=True, scale=dpi/72))
│   └── converter.convert(pdf_path)
│
├── doc.export_to_markdown()       → ParsedDoc.markdown  (full text + inline tables)
├── doc.tables[*].export_to_markdown() → ParsedDoc.tables_md  (standalone Markdown tables)
└── doc.pages[n].image.pil_image.save(…) → ParsedDoc.page_images  [Path, …]
```

`ParsedDoc` is a plain dataclass — no framework coupling.

---

### `rag/indexer.py` — dual vector store

#### `TextIndex` — FAISS + sentence-transformers

```
TextIndex(model_name)
│
├── add(chunks)
│   ├── SentenceTransformer.encode(texts, normalize_embeddings=True)
│   └── faiss.IndexFlatIP.add(embeddings)        # inner-product = cosine on normalised vecs
│
├── search(query, k) → [(score, chunk_dict), …]
│   └── index.search(q_embedding, k)
│
├── save(path) / load(path)
│   ├── faiss.write_index / read_index
│   └── pickle chunks list
```

#### `VisualIndex` — Byaldi + ColQwen2

```
VisualIndex(model_name)
│
├── index(pdf_dir, index_dir)
│   └── RAGMultiModalModel.from_pretrained(colqwen_model)
│       └── model.index(input_path=pdf_dir, index_name="visual_index")
│
├── load(index_dir)
│   └── RAGMultiModalModel.from_index(…)
│
└── search(query, k) → [{"score", "doc_id", "page_num", "base64"}, …]
    └── model.search(query, k, return_base64_results=True)
```

`VisualIndex` is lazy-imported — if `byaldi` is not installed the pipeline
falls back to text-only retrieval without raising an error.

---

### `rag/pipeline.py` — RAGPipeline

The central orchestrator.

#### `index_documents(pdf_dir)`

```
for each PDF in pdf_dir:
    parse_pdf(pdf_path) → ParsedDoc
    TextIndex.add([{text: markdown + tables, pdf_id, page_num}])

VisualIndex.index(pdf_dir, index_dir)     # no-op if byaldi unavailable
```

#### `query(question) → dict`

```
query(question)
│
├── TextIndex.search(question, k)      → [(score, chunk), …]
├── VisualIndex.search(question, k)    → [{score, base64, …}, …]
│
├── best_score = max(text scores + visual scores)
│
├── [best_score < threshold?]
│   └── YES → return NO_INFO_MSG   ← no LLM call
│
├── build multimodal content list
│   ├── {"type": "text", "text": context + question}
│   └── {"type": "image_url", "image_url": …}  × up to 2 page images
│
└── client.chat.completions.create(model, messages, max_tokens=512)
    └── return {answer, retrieved_chunks, visual_results, best_score}
```

The similarity threshold gate prevents hallucination on out-of-scope queries
and avoids unnecessary LLM calls.

---

### `evaluate.py` — evaluation loop

```
run(csv_path, pdf_dir, config_path, out_path)
│
├── RAGPipeline(config_path).index_documents(pdf_dir)
│
├── for each row in CSV:
│   └── pipeline.query(question)
│       └── record {question, ground_truth, predicted, retrieved_chunks, best_score}
│
├── bert_score(hyps, refs, lang="en")   → P, R, F1 per sample
│   └── append bert_f1 to each record
│
├── print avg BERTScore F1
└── write JSON log to out_path
```

**BERTScore F1** is the primary metric — it captures semantic similarity
without requiring exact string matches, which matters for free-text answers.
Instances where `best_score < threshold` (i.e. visual-only queries where text
retrieval fails) can be identified in the log by their low `best_score` value.

---

## End-to-End Data Flow

```
config.yaml
  engine / model / quantization / embedding_model / colqwen_model
         │
         ├─────────────────────────────────────────────────────────► LLM server
         │                                                           (Ollama / llama.cpp / vLLM)
         ▼
src/part2/example_data/*.pdf
         │
         ▼ parse_pdf (Docling)
    ParsedDoc {markdown, tables_md, page_images}
         │
         ├──► TextIndex (sentence-transformers + FAISS)
         │
         └──► VisualIndex (ColQwen2 + Byaldi)
                        │
    question ──────────►├──► TextIndex.search()  →  top-K text chunks
                        └──► VisualIndex.search() → top-K page images + scores
                                    │
                          [best_score < threshold?]
                          NO  ──► multimodal LLM prompt ──► answer
                          YES ──► "The document does not contain information…"
                                    │
                         evaluate.py (BERTScore F1 vs ground-truth CSV)
                                    │
                              eval_results.json
```

---

## GPU Memory Budget (T4 16 GB)

| Component         | Allocation | Strategy                                          |
|-------------------|------------|---------------------------------------------------|
| ColQwen2-2B       | ~3–4 GB    | Load via Byaldi; add `BitsAndBytesConfig` for 4-bit to halve this |
| Gemma 3 4B (GGUF) | ~2.5 GB    | Served by Ollama using `q4_K_M` tag               |
| FAISS index       | < 100 MB   | CPU-side; no GPU memory used                      |
| Page images       | < 500 MB   | Capped at 300 DPI; sent as base64 in prompt       |
| **Total**         | **~7 GB**  | Comfortable headroom on 16 GB                     |

---

## Adding a New Embedding Model

1. Add the HuggingFace model id to the `embedding_model` field in the relevant
   config file under [config/](config/).
2. No code change required — `TextIndex` accepts any `sentence-transformers`
   compatible model name.

## Adding a New Inference Engine

1. Add its `base_url` to `ENGINE_BASE_URLS` in [engines/factory.py](engines/factory.py).
2. Add an example config file under [config/](config/).
3. Add the new entry to `ENGINE_CONFIGS` in [tests/part2/conftest.py](../../tests/part2/conftest.py) —
   all parametrised tests will automatically cover it.
