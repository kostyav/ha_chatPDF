# ha_chatPDF


## PART1

### 1. Install dependencies

```bash
pip install -r requirements.txt   # runtime + test deps
pip install -e .                  # make src.part1 importable
```

> **Note — HuggingFace authentication:** `google/gemma-3-1b-it` and the llama.cpp GGUF counterpart are gated models. You must accept their terms on HuggingFace and authenticate before models can be downloaded:
> ```bash
> huggingface-cli login --token <your_token>
> ```
> Generate a token at https://huggingface.co/settings/tokens, then accept the model terms at https://huggingface.co/google/gemma-3-1b-it (used by vLLM) and https://huggingface.co/unsloth/gemma-3-1b-it-GGUF (used by llama.cpp).

> **Note — numpy 2.x compatibility:** `requirements.txt` pins `scipy>=1.13`, `scikit-learn>=1.5`, and `pandas>=2.2`. Older versions of these packages were compiled against numpy 1.x and will cause an `ImportError` (e.g. `cannot import name 'Inf' from numpy`) when vLLM is used. If you manage these packages outside of this file, ensure they meet the minimum versions above.

### 2. Configure the engine

Edit `src/part1/config/config.yaml` to select the engine, model, and quantization:

```yaml
engine: ollama          # ollama | llamacpp | vllm
model: gemma:3-1b-it
quantization: q4_0
```

Per-engine example configs are in `src/part1/config/`.

### 3. Start the inference server

**Option A — native (engine must be installed on the host):**

```bash
./src/part1/deploy.sh                        # uses config.yaml
./src/part1/deploy.sh src/part1/config/config.ollama.yaml    # explicit config
./src/part1/deploy.sh src/part1/config/config.llamacpp.yaml
./src/part1/deploy.sh src/part1/config/config.vllm.yaml
```

**Option B — Docker:**

```bash
./src/part1/deploy.sh src/part1/config/config.ollama.yaml   --docker
./src/part1/deploy.sh src/part1/config/config.llamacpp.yaml --docker
./src/part1/deploy.sh src/part1/config/config.vllm.yaml     --docker
```

Engine endpoints once running:

| Engine   | URL                          |
|----------|------------------------------|
| Ollama   | http://localhost:11434/v1    |
| llama.cpp| http://localhost:8080/v1     |
| vLLM     | http://localhost:8000/v1     |

### 4. Run tests

```bash
pytest -m "not integration"   # unit tests — no server needed
pytest -m integration         # Hello World + per-engine checks (server must be running)
```

