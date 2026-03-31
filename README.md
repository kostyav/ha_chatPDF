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


## PART2

### 1. Install dependencies

```bash
pip install -r src/part2/requirements.txt
pip install -e .                            # make src.part2 importable
```

### 2. Configure the engine

Edit `src/part2/config/config.yaml`:

```yaml
engine: ollama          # ollama | llamacpp | vllm
model: gemma3:4b
quantization: q4_K_M
embedding_model: sentence-transformers/all-MiniLM-L6-v2
```

Per-engine configs are in `src/part2/config/`.

### 3. Pull the Ollama model

```bash
docker compose -f src/part2/docker-compose.yml up -d ollama
docker compose -f src/part2/docker-compose.yml exec ollama ollama pull gemma3:4b
```

### 4. Copy PDFs into the shared volume

```bash
docker volume create rag-part2_pdf-data
docker run --rm \
  -v rag-part2_pdf-data:/data/pdfs \
  -v $(pwd)/src/part2/example_data:/src:ro \
  alpine sh -c "ls /src/*.pdf | head -2 | xargs -I{} cp -v {} /data/pdfs/"
```

### 5. Start all services

```bash
docker compose -f src/part2/docker-compose.yml up --build
```

### 6. Run a query

`main.py` only handles indexing. Use `query.py` for interactive querying:

```bash
# Interactive REPL
docker compose -f src/part2/docker-compose.yml run --rm -it orchestrator python query.py

# Single question
docker compose -f src/part2/docker-compose.yml run --rm orchestrator \
  python query.py --question "Which section describes Fig. 4?"
```

### 7. Run tests

```bash
pytest tests/part2 -m "not integration"   # unit tests — no server needed
pytest tests/part2 -m integration         # requires live Ollama server
```

---

## Lightning.ai — preventing container rebuilds on restart

On Lightning.ai the Docker layer cache lives in ephemeral instance storage and
is wiped on every machine restart, forcing a full rebuild. To avoid this, build
once and save the images to the **persistent** workspace filesystem.

### First time setup (or after code changes)

```bash
make build   # builds all four service images and saves them to .docker_images/
make pull    # saves redis, qdrant, and ollama images too
```

> Add `.docker_images/` to `.gitignore` — the tar files are large:
> ```bash
> echo ".docker_images/" >> .gitignore
> ```

### Every subsequent restart (no rebuild)

```bash
make up      # loads images from .docker_images/*.tar, then starts all services
```

`make up` automatically falls back to a full build if no cached images are found.

### Available make targets

| Target       | Description                                              |
|--------------|----------------------------------------------------------|
| `make build` | Build service images and save to `.docker_images/`       |
| `make pull`  | Pull and save infrastructure images (Redis, Qdrant, Ollama) |
| `make load`  | Load all saved `.tar` images into Docker (without starting) |
| `make up`    | Load cached images (or build) then start all services    |
| `make down`  | Stop and remove containers                               |
