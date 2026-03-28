# Tests

## Setup

```bash
pip install -r requirements.txt   # openai, pyyaml, pytest
pip install -e .                  # make src.part1 importable as a package
```

> **HuggingFace authentication required for llamacpp and vllm integration tests.**
> The configured models (`google/gemma-3-1b-it`, `bartowski/gemma-3-1b-it-GGUF`) are gated.
> Without a valid token the fixtures skip those tests with an actionable message.
> To enable them:
> ```bash
> huggingface-cli login --token <your_token>
> ```
> Get a token at https://huggingface.co/settings/tokens and accept model terms at https://huggingface.co/google/gemma-3-1b-it.

---

## Layout

```
tests/
└── part1/
    ├── conftest.py               ← shared fixtures and parametrize matrix
    ├── test_hello_world.py       ← deliverable verification (live server)
    ├── test_configurations.py    ← factory unit tests + per-engine integration tests
    └── test_engine_install.py    ← deploy.sh structure checks + engine availability
```

---

## Markers

Tests are tagged with markers to control which subset runs:

| Marker        | Meaning                                                  | Requires          |
|---------------|----------------------------------------------------------|-------------------|
| _(none)_      | Pure unit test — reads files or calls in-process code    | nothing           |
| `integration` | Sends real HTTP requests to an inference server endpoint | live server       |
| `install`     | Runs `pip install` — mutates the Python environment      | network + disk    |

Markers are registered in [`pyproject.toml`](../pyproject.toml).

---

## Test files

### `conftest.py`

Shared state consumed by all test files in `tests/part1/`.

| Symbol           | Type    | Description                                                              |
|------------------|---------|--------------------------------------------------------------------------|
| `ENGINE_CONFIGS` | list    | `[(config_path, engine_name, port), ...]` for all three engines          |
| `default_config_path` | fixture | Path to the active `src/part1/config/config.yaml`                  |

---

### `test_hello_world.py` — deliverable verification

Single `@pytest.mark.integration` test. Reads the active `config.yaml`, obtains a client via `get_client()`, sends `"Say exactly: Hello World"` and asserts the response is non-empty.

```
test_hello_world
└── get_client(config.yaml) → (OpenAI client, model)
    └── client.chat.completions.create(...)
        └── assert response non-empty
```

**Requires** the engine named in `config.yaml` to be running.

---

### `test_configurations.py` — factory correctness

#### Unit tests (`TestLoadConfig`, `TestGetClient`) — no server needed

| Test | What it asserts |
|------|-----------------|
| `test_loads_engine` | `config.yaml` contains a recognised engine name |
| `test_loads_model` | `config.yaml` has a non-empty model field |
| `test_engine_in_each_config` ×3 | each per-engine config file declares the right engine |
| `test_client_base_url` ×3 | `get_client()` returns an `OpenAI` client with the correct port in `base_url` |
| `test_unknown_engine_raises` | `get_client()` raises `ValueError` for an unknown engine name |
| `test_model_returned` ×3 | `get_client()` returns a non-empty model string |

#### Integration tests — live server required

| Test | What it asserts |
|------|-----------------|
| `test_chat_completion_per_engine` ×3 | single-word completion is non-empty |
| `test_model_list_per_engine` ×3 | `/v1/models` returns at least one model |

Call chain for all tests in this file:

```
get_client(config_path)
├── load_config(path) → dict        # reads YAML
└── OpenAI(base_url=ENGINE_URL)     # constructs client
    └── (integration only) client.chat.completions.create(...)
```

---

### `test_engine_install.py` — engine availability

#### Structure checks — no install or server needed

| Test | What it asserts |
|------|-----------------|
| `test_install_function_defined_in_deploy_sh` ×3 | `install_ollama/llamacpp/vllm()` are defined in `deploy.sh` |
| `test_engine_case_in_deploy_sh` ×3 | each engine has a `case` block in `deploy.sh` |
| `test_install_functions_called_on_missing_binary` | each install function appears at least twice (defined + called) |

#### Ollama availability checks — requires ollama installed

| Test | What it asserts |
|------|-----------------|
| `test_ollama_binary_available` | `ollama` is on `PATH` |
| `test_ollama_version` | `ollama --version` exits 0 |

#### llama-cpp-python checks — skip if not installed, install with `-m install`

| Test | Marker | What it asserts |
|------|--------|-----------------|
| `test_llamacpp_pip_install` | `install` | `pip install llama-cpp-python[server]` exits 0 |
| `test_llamacpp_module_importable` | — | `import llama_cpp` succeeds (skips if absent) |
| `test_llamacpp_server_invocable` | — | `python -m llama_cpp.server --help` exits 0 (skips if absent) |

#### vLLM checks — skip if not installed, install with `-m install`

| Test | Marker | What it asserts |
|------|--------|-----------------|
| `test_vllm_pip_install` | `install` | `pip install vllm` exits 0 |
| `test_vllm_module_importable` | — | `import vllm` succeeds (skips if absent) |
| `test_vllm_server_invocable` | — | `python -m vllm.entrypoints.openai.api_server --help` exits 0 (skips if absent) |

---

## How to run

### Unit tests only (no server, no install)

```bash
pytest -m "not integration and not install"
```

### Unit tests + deploy.sh structure checks + ollama availability

```bash
pytest tests/part1/test_engine_install.py tests/part1/test_configurations.py \
       -m "not integration and not install"
```

### Install llama-cpp-python and verify it

```bash
pytest tests/part1/test_engine_install.py -m install -k llamacpp
pytest tests/part1/test_engine_install.py -k llamacpp   # then verify
```

### Install vLLM and verify it

```bash
pytest tests/part1/test_engine_install.py -m install -k vllm
pytest tests/part1/test_engine_install.py -k vllm        # then verify
```

### Full integration suite (all three engines must be running)

```bash
# Start all three servers first:
./src/part1/deploy.sh src/part1/config/config.ollama.yaml
./src/part1/deploy.sh src/part1/config/config.llamacpp.yaml
./src/part1/deploy.sh src/part1/config/config.vllm.yaml

pytest -m integration
```

### Hello World against the active engine

```bash
# Edit src/part1/config/config.yaml to select the engine, then:
./src/part1/deploy.sh
pytest tests/part1/test_hello_world.py -m integration -s   # -s prints the model reply
```

### Everything at once

```bash
pytest                          # all markers (skips gracefully when server/engine absent)
pytest -v --tb=short            # verbose with compact tracebacks
```
