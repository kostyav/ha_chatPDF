# Architecture — Part 1: Local LLM Deployment

## Overview

The system deploys a local LLM inference server and exposes it through a thin
abstraction layer.  The key insight is that **all three supported engines
(Ollama, llama.cpp, vLLM) implement the OpenAI-compatible REST API** at
`/v1/chat/completions`.  The entire abstraction therefore reduces to a single
factory that maps an engine name to a `base_url` and returns a configured
`openai.OpenAI` client — no bespoke HTTP code required.

---

## File Layout

```
src/part1/
├── ARCHITECTURE.md          ← this file
├── README.md                ← task specification
├── requirements.txt         ← openai, pyyaml
│
├── config/                  ← runtime configuration
│   ├── config.yaml          ← ACTIVE config (edit this to switch engine/model)
│   ├── config.ollama.yaml   ← example: Ollama engine
│   ├── config.llamacpp.yaml ← example: llama.cpp engine
│   └── config.vllm.yaml     ← example: vLLM engine
│
├── engines/                 ← Python abstraction layer
│   ├── __init__.py          ← re-exports get_client, load_config
│   └── factory.py           ← core factory (~20 LOC)
│
├── deploy.sh                ← shell deployment script (native + Docker modes)
└── docker-compose.yml       ← Docker Compose deployment (one profile per engine)

tests/part1/
├── __init__.py
├── conftest.py              ← shared fixtures and parametrize matrix
├── test_hello_world.py      ← deliverable: "Hello World" verification
└── test_configurations.py  ← unit + integration tests across all engines
```

---

## Configuration Schema

Every config file is a plain YAML document with three keys:

```yaml
engine: ollama           # one of: ollama | llamacpp | vllm
model: gemma:3-1b-it     # engine-specific identifier (see notes below)
quantization: q4_0       # optional — interpretation varies by engine
```

### How `model` and `quantization` are interpreted per engine

| Engine    | `model` value                        | `quantization` effect                         |
|-----------|--------------------------------------|-----------------------------------------------|
| ollama    | Ollama tag, e.g. `gemma:3-1b-it`     | Appended as tag suffix: `gemma:3-1b-it:q4_0`  |
| llamacpp  | Local `.gguf` file path              | Baked into the filename; field is ignored      |
| vllm      | HuggingFace model id                 | Passed as `--quantization` CLI flag            |

---

## Python Abstraction — `engines/factory.py`

### Public surface

```
load_config(path=None) -> dict
get_client(config_path=None) -> (OpenAI, str)
```

### Call hierarchy

```
get_client(config_path)
│
├── load_config(config_path)          # reads YAML → dict
│   └── yaml.safe_load(file)
│
├── ENGINE_BASE_URLS[engine]          # dict lookup → base_url string
│   ├── "ollama"   → http://localhost:11434/v1
│   ├── "llamacpp" → http://localhost:8080/v1
│   └── "vllm"     → http://localhost:8000/v1
│
└── OpenAI(base_url=..., api_key="none")   # openai SDK client
    └── returns (client, model_name)
```

`get_client` is the **single entry point** for all callers (tests, scripts,
application code).  It is idempotent and side-effect-free — it does not start
servers or pull models.

---

## Deployment — `deploy.sh`

### Invocation

```
./src/part1/deploy.sh [config.yaml] [--docker]
```

`config.yaml` defaults to `src/part1/config/config.yaml`.

### Execution flow

```
deploy.sh
│
├── _cfg()   ← inline python3 snippet reads YAML fields (engine, model, quantization)
│
├── [--docker flag present?]
│   └── YES → docker compose --profile $ENGINE up -d
│               └── delegates to docker-compose.yml (see below)
│
└── NO (native mode)
    ├── engine = ollama
    │   ├── ollama serve &          # background daemon
    │   ├── sleep 2                 # wait for daemon
    │   └── ollama pull $MODEL[:$QUANT]
    │
    ├── engine = llamacpp
    │   └── llama-server --model $MODEL --port 8080 --host 0.0.0.0 &
    │
    └── engine = vllm
        └── python -m vllm.entrypoints.openai.api_server \
                --model $MODEL [--quantization $QUANT] &
```

---

## Deployment — `docker-compose.yml`

One service per engine, activated via **Docker Compose profiles** so only the
selected engine starts.

```
docker-compose.yml
├── service: ollama    (profile: ollama)    → image: ollama/ollama:latest        :11434
├── service: llamacpp  (profile: llamacpp)  → image: ghcr.io/ggerganov/llama.cpp :8080
└── service: vllm      (profile: vllm)      → image: vllm/vllm-openai:latest     :8000
```

`deploy.sh --docker` sets `MODEL` and `QUANTIZATION` env vars before calling
`docker compose --profile $ENGINE up -d`, so the compose file stays static.

---

## Tests — `tests/part1/`

### conftest.py

Provides two shared fixtures used by both test files:

- `default_config_path` — fixture returning path to the active `config.yaml`
- `ENGINE_CONFIGS` — module-level list of `(config_path, engine, port)` tuples
  used for parametrized tests across all three engines

### test_hello_world.py (deliverable verification)

```
test_hello_world(default_config_path)
│
├── get_client(default_config_path)   → (client, model)
└── client.chat.completions.create(...)
    └── assert response is non-empty
```

Runs against whichever engine is set in `config.yaml`.

### test_configurations.py

Two test classes and two integration test functions:

```
TestLoadConfig
├── test_loads_engine            — config.yaml has a known engine key
├── test_loads_model             — config.yaml has a non-empty model key
└── test_engine_in_each_config   — parametrized × 3 engine configs

TestGetClient
├── test_client_base_url         — parametrized × 3: correct port in base_url
├── test_unknown_engine_raises   — ValueError on bad engine name
└── test_model_returned          — parametrized × 3: non-empty model string

@pytest.mark.integration (require a live server)
├── test_chat_completion_per_engine  — parametrized × 3: non-empty LLM reply
└── test_model_list_per_engine       — parametrized × 3: /v1/models returns ≥1
```

Run only unit tests (no server needed):

```bash
pytest tests/part1 -m "not integration"
```

Run integration tests against a live server:

```bash
./src/part1/deploy.sh                 # start the configured engine
pytest tests/part1 -m integration
```

---

## End-to-End Data Flow

```
config.yaml
    │  engine / model / quantization
    ▼
deploy.sh ──────────────────────────────► inference server (Ollama / llama.cpp / vLLM)
                                                │
                                                │  OpenAI-compatible REST API
                                                │  POST /v1/chat/completions
                                                │
engines/factory.py → get_client()              │
    └── openai.OpenAI(base_url=ENGINE_URL) ────►│
                        ◄──── { choices[0].message.content }
                                                │
tests/part1/test_hello_world.py ───────────────┘
    assert reply is non-empty
```

---

## Adding a New Engine

1. Add its `base_url` to `ENGINE_BASE_URLS` in [engines/factory.py](engines/factory.py).
2. Add a `case` block to [deploy.sh](deploy.sh).
3. Add a service with the matching profile name to [docker-compose.yml](docker-compose.yml).
4. Add an example config file under [config/](config/).
5. Add the new entry to `ENGINE_CONFIGS` in [tests/part1/conftest.py](../../tests/part1/conftest.py) — all parametrized tests will automatically cover it.
