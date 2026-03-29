# Tests — Part 2: Multimodal RAG System

## Setup

```bash
pip install -r src/part2/requirements.txt
pip install -e .                           # make src.part2 importable as a package
pip install pytest bert-score              # test dependencies
```

> **HuggingFace authentication required for llamacpp and vllm integration tests.**
> The models (`google/gemma-3-4b-it`, `unsloth/gemma-3-4b-it-GGUF`) are gated.
> To enable them:
> ```bash
> huggingface-cli login --token <your_token>
> ```
> Accept model terms at https://huggingface.co/google/gemma-3-4b-it.

---

## Layout

```
tests/part2/
├── conftest.py               ← shared fixtures, server lifecycle, ENGINE_CONFIGS
├── test_config.py            ← config loading and engine factory (all 3 engines)
├── test_parser.py            ← Docling PDF parser (mocked + real-PDF integration)
├── test_pipeline.py          ← RAG pipeline: threshold gating, LLM call, image attachment
├── test_configurations.py    ← pipeline init + live inference per engine/model/quantization
└── test_evaluate.py          ← evaluation script: BERTScore output, JSON logging
```

---

## Markers

| Marker        | Meaning                                                                | Requires          |
|---------------|------------------------------------------------------------------------|-------------------|
| _(none)_      | Pure unit test — in-process code only, all heavy deps mocked           | nothing           |
| `integration` | Sends real HTTP / model requests; may parse actual PDFs                | live server / GPU |

Markers are registered in [`pyproject.toml`](../../pyproject.toml).

---

## Test files

### `conftest.py` — shared infrastructure

| Symbol                  | Type     | Description                                                            |
|-------------------------|----------|------------------------------------------------------------------------|
| `ENGINE_CONFIGS`        | list     | `[(config_path, engine_name, port), ...]` for all three engines        |
| `ENGINE_SERVER_FIXTURE` | dict     | Maps engine name → session-scoped server fixture name                  |
| `default_config_path`   | fixture  | Path to the active `src/part2/config/config.yaml`                     |
| `EXAMPLE_DIR`           | Path     | `src/part2/example_data/` (10 PDFs + ground-truth CSV)                 |
| `ollama_server`         | fixture  | Starts Ollama on :11434 and pulls the configured model (session-scoped) |
| `llamacpp_server`       | fixture  | Downloads GGUF from HuggingFace and serves on :8080 (session-scoped)   |
| `vllm_server`           | fixture  | Downloads HF model and serves on :8000 (session-scoped)                |

---

### `test_config.py` — config loading and engine factory

#### `TestLoadConfig`

| Test | What it asserts |
|------|-----------------|
| `test_default_has_required_keys` | default `config.yaml` has `engine`, `model`, `embedding_model` |
| `test_default_engine_is_valid` | default engine is one of the three known engines |
| `test_engine_matches_file` ×3 | each per-engine config declares the matching engine name |
| `test_model_non_empty` ×3 | `model` field is non-empty in every config |
| `test_quantization_key_present` ×3 | `quantization` key is present (may be empty string) |
| `test_retriever_section` ×3 | `retriever.top_k ≥ 1` and `0 < similarity_threshold < 1` |

#### `TestGetClient`

| Test | What it asserts |
|------|-----------------|
| `test_returns_openai_client` ×3 | returns `openai.OpenAI` with the correct port in `base_url` |
| `test_model_non_empty` ×3 | returned model string is non-empty |
| `test_unknown_engine_raises` | `ValueError` raised for an unknown engine name |

---

### `test_parser.py` — PDF parsing

All unit tests mock `DocumentConverter` so Docling is never actually called.

| Test | What it asserts |
|------|-----------------|
| `test_parse_returns_parseddoc` | return type is `ParsedDoc` |
| `test_parse_markdown_content` | `ParsedDoc.markdown` contains the mocked Markdown text |
| `test_parse_extracts_tables` | `ParsedDoc.tables_md` populated from `doc.tables` |
| `test_parse_saves_page_images` | `PIL.Image.save()` called for each page with an image |
| `test_output_dir_created` | output directory is created when it does not exist |

#### Integration test

| Test | Marker | What it asserts |
|------|--------|-----------------|
| `test_parse_real_pdf` | `integration` | Docling parses `23870758.pdf`; Markdown is non-empty |

---

### `test_pipeline.py` — RAG pipeline logic

The LLM client is always mocked via `patch("src.part2.rag.pipeline.get_client")`.

| Test | What it asserts |
|------|-----------------|
| `test_pipeline_init` | `_top_k > 0`, `0 < _threshold < 1` |
| `test_query_empty_index_returns_no_info` | empty index → `NO_INFO_MSG` returned, no LLM call |
| `test_query_below_threshold` | score below threshold → `NO_INFO_MSG`, no LLM call |
| `test_query_above_threshold_calls_llm` | score above threshold → LLM called, answer returned |
| `test_query_result_has_required_keys` | result always contains `answer`, `retrieved_chunks`, `visual_results`, `best_score` |
| `test_retrieved_chunks_contain_metadata` | each chunk has `pdf_id` and `score` |
| `test_visual_hits_attached_to_prompt` | base64 images from visual hits appear as `image_url` parts in the LLM message |

#### Integration test

| Test | Marker | What it asserts |
|------|--------|-----------------|
| `test_index_and_query_live` | `integration` | Full index + query cycle against a live Ollama server |

---

### `test_configurations.py` — per-engine/model/quantization coverage

#### Unit tests (no server)

| Test | What it asserts |
|------|-----------------|
| `test_config_retriever_settings` ×3 | `top_k ≥ 1`, `0 < similarity_threshold < 1` per engine config |
| `test_config_parser_settings` ×3 | `parser.dpi > 0` per engine config |
| `test_config_embedding_model` ×3 | `embedding_model` key is non-empty per engine config |
| `test_pipeline_init_each_engine` ×3 | `RAGPipeline` initialises without error for every engine config |
| `test_pipeline_uses_configured_embedding_model` ×3 | pipeline loads the embedding model named in the config |

#### Integration tests (live server per engine)

| Test | What it asserts |
|------|-----------------|
| `test_chat_completion_per_engine` ×3 | engine returns a non-empty chat completion |
| `test_rag_query_per_engine` ×3 | full index + query returns a result dict for each engine |

---

### `test_evaluate.py` — evaluation script

All tests mock `RAGPipeline` to avoid heavy model loading.

| Test | What it asserts |
|------|-----------------|
| `test_run_returns_records_and_f1` | `run()` returns a list of records and a float F1 in `[0, 1]` |
| `test_run_writes_json_output` | JSON output file contains `avg_bert_f1` and per-record results |
| `test_run_each_record_has_bert_f1` | every record has a `bert_f1` key in `[0, 1]` |
| `test_run_no_info_answers_get_low_score` | `NO_INFO_MSG` responses score below 0.9 BERTScore F1 |
| `test_run_skips_json_write_when_out_path_none` | passing `out_path=None` returns results without writing a file |

---

## How to run

### Unit tests only (no server, no GPU)

```bash
pytest tests/part2 -m "not integration"
```

### Specific file

```bash
pytest tests/part2/test_config.py -v
pytest tests/part2/test_parser.py -v
pytest tests/part2/test_pipeline.py -v
pytest tests/part2/test_evaluate.py -v
```

### Integration tests against Ollama only

```bash
ollama serve &
ollama pull gemma3:4b
pytest tests/part2 -m integration -k "ollama"
```

### Full integration suite (all three engines)

```bash
# Start all three servers first, then:
pytest tests/part2 -m integration
```

### Verbose with short tracebacks

```bash
pytest tests/part2 -v --tb=short
```

### Everything at once (skips gracefully when server absent)

```bash
pytest tests/part2
```
