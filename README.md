# ha_chatPDF


## PART1

### 1. Install dependencies

```bash
pip install -r requirements.txt   # runtime + test deps
pip install -e .                  # make src.part1 importable
```

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

