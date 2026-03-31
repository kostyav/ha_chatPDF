# CLAUDE.md — Part 1: LLM Inference Engine Abstraction

## Purpose

Deploys a local LLM across three interchangeable inference backends (Ollama, llama.cpp, vLLM), all exposed via OpenAI-compatible REST API. Caller code never changes when switching engines — only the YAML config does.

---

## Directory Structure

```
src/part1/
├── CLAUDE.md                  # This file
├── ARCHITECTURE.md            # Design rationale and data flow
├── README.md                  # Task specification
├── requirements.txt           # openai>=1.0, pyyaml>=6.0
├── deploy.sh                  # Launches engine natively or via Docker
├── docker-compose.yml         # Profiles: ollama | llamacpp | vllm
├── config/
│   ├── config.yaml            # Active config (engine, model, quantization)
│   ├── config.ollama.yaml     # Example: Ollama + gemma3:1b
│   ├── config.llamacpp.yaml   # Example: llama.cpp + GGUF path
│   └── config.vllm.yaml       # Example: vLLM + HF model ID
└── engines/
    ├── __init__.py            # Re-exports get_client, load_config
    └── factory.py             # Core abstraction (~30 LOC)
```

---

## Core Abstraction: `engines/factory.py`

The entire Python layer is ~30 lines. Two public functions:

```python
load_config(path=None) -> dict        # Reads YAML, returns dict
get_client(config_path=None) -> (OpenAI, str)  # Returns (client, model_name)
```

**Engine-to-URL mapping** (lookup table, no if/else):
```python
ENGINE_BASE_URLS = {
    "ollama":   "http://localhost:11434/v1",
    "llamacpp": "http://localhost:8080/v1",
    "vllm":     "http://localhost:8000/v1",
}
```

**Key implementation details:**
- Uses official OpenAI Python SDK with `base_url` override — no custom HTTP code
- `api_key="none"` — local servers don't validate keys; dummy satisfies SDK init
- `DEFAULT_CONFIG` resolved relative to the package file, not CWD
- Returns `(client, model_name)` tuple — caller uses both for `chat.completions.create()`

**Usage pattern:**
```python
from src.part1.engines import get_client
client, model = get_client()  # reads config.yaml
response = client.chat.completions.create(model=model, messages=[...])
```

---

## Configuration Schema

Each YAML config has exactly three keys:

| Key | Description | Example |
|-----|-------------|---------|
| `engine` | Which backend: `ollama`, `llamacpp`, `vllm` | `ollama` |
| `model` | Engine-specific model identifier | `gemma3:1b` |
| `quantization` | Optional quant spec (engine-dependent) | `""`, `q4_K_M`, `bitsandbytes` |

**Active config:** `config/config.yaml` (currently: `engine: ollama`, `model: gemma3:1b`)

**Quantization is handled differently per engine:**
- Ollama: appended to model tag → `gemma3:1b-q4_K_M`
- llama.cpp: baked into GGUF filename → `gemma-3-1b-it-Q4_K_M.gguf`
- vLLM: passed as CLI flag → `--quantization bitsandbytes`

---

## Deployment

### Native (default)
```bash
./deploy.sh                                  # uses config/config.yaml
./deploy.sh config/config.ollama.yaml        # explicit config
```
Script auto-installs missing engine dependencies, downloads models, starts server.

### Docker
```bash
./deploy.sh config/config.ollama.yaml --docker    # starts only the ollama profile
```
Uses `docker compose --profile $ENGINE up -d`. Only one engine activates at a time.

### Ports
| Engine | Port |
|--------|------|
| Ollama | 11434 |
| llama.cpp | 8080 |
| vLLM | 8000 |

---

## Docker Compose Profiles

Three services, each behind a profile (`ollama`, `llamacpp`, `vllm`). Only the selected profile activates. Key service details:

- **ollama**: `ollama/ollama:latest`, port 11434, pulls model via entrypoint
- **llamacpp**: `ghcr.io/ggerganov/llama.cpp:server`, port 8080, GGUF from `./models/`
- **vllm**: `vllm/vllm-openai:latest`, port 8000, HF cache volume, GPU reserved

---

## `deploy.sh` Internals

Config extraction via inline Python (avoids `yq` dependency):
```bash
_cfg() { python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('$1',''))"; }
```

**Native engine startup:**
- Ollama: `ollama serve &` → `ollama pull $MODEL` (with optional quant suffix)
- llama.cpp: Downloads GGUF via `huggingface_hub`, runs `python -m llama_cpp.server`
- vLLM: `huggingface_hub.snapshot_download()` then `python -m vllm.entrypoints.openai.api_server`

---

## Tests

Located at `tests/part1/`. Two markers:
- No marker: unit tests, no server required (mock or fixture)
- `@pytest.mark.integration`: requires a running engine server

```bash
pytest tests/part1 -m "not integration"   # unit tests only
pytest tests/part1 -m integration          # needs running server
```

Tests are parametrized across all three engines to verify identical API behavior.

---

## Adding a New Engine

1. Add entry to `ENGINE_BASE_URLS` in `factory.py`
2. Add a new Docker service with the new profile in `docker-compose.yml`
3. Add native startup block in `deploy.sh`
4. Add example `config/config.<engine>.yaml`

No other code changes required.

---

## Dependencies

```
openai>=1.0    # OpenAI SDK — client for all engines
pyyaml>=6.0   # YAML config parsing
```

Engine runtimes are installed by `deploy.sh` as needed (not in requirements.txt).
