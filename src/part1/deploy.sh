#!/usr/bin/env bash
# Usage: ./deploy.sh [config.yaml] [--docker]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config/config.yaml}"
DOCKER="${2:-}"

# Parse config via python (avoids a yq dependency)
_cfg() { python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('$1',''))"; }
ENGINE=$(_cfg engine)
MODEL=$(_cfg model)
QUANT=$(_cfg quantization)

echo "Engine: $ENGINE | Model: $MODEL | Quantization: ${QUANT:-none}"

# ── Docker mode ────────────────────────────────────────────────────────────────
if [[ "$DOCKER" == "--docker" ]]; then
    MODEL="$MODEL" QUANTIZATION="$QUANT" \
        docker compose -f "$SCRIPT_DIR/docker-compose.yml" --profile "$ENGINE" up -d
    echo "Started $ENGINE container. Endpoint ready shortly."
    exit 0
fi

# ── Dependency check helpers ───────────────────────────────────────────────────
need() {
  local cmd="$1" install_hint="$2"
  if ! command -v "$cmd" &>/dev/null; then
    echo "ERROR: '$cmd' not found."
    echo "Install: $install_hint"
    exit 1
  fi
}

install_ollama() {
  echo "Ollama not found. Installing via official script..."
  curl -fsSL https://ollama.com/install.sh | sh
}

install_llamacpp() {
  echo "llama-cpp-python not found. Installing via pip..."
  pip install --quiet 'llama-cpp-python[server]'
}

install_vllm() {
  echo "vLLM not found. Installing via pip..."
  pip install --quiet vllm
}

# ── Model download helpers ─────────────────────────────────────────────────────

download_gguf() {
  # Download a single file from a HuggingFace repo into the directory of $3.
  local hf_repo="$1" hf_file="$2" dest="$3"
  if [[ -f "$dest" ]]; then
    echo "Model already present: $dest"; return 0
  fi
  echo "Downloading $hf_repo/$hf_file → $dest ..."
  python3 -c "
from huggingface_hub import hf_hub_download
import shutil, pathlib
src = hf_hub_download(repo_id='$hf_repo', filename='$hf_file')
dst = pathlib.Path('$dest')
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(src, dst)
print('Saved:', dst)
"
}

download_hf_model() {
  # Pre-download a full HuggingFace model repo into the HF cache (used by vLLM).
  local hf_model="$1"
  echo "Pre-downloading HuggingFace model: $hf_model ..."
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$hf_model')
print('Download complete.')
"
}

# ── Native mode ────────────────────────────────────────────────────────────────
case "$ENGINE" in
  ollama)
    command -v ollama &>/dev/null || install_ollama
    ollama serve &>/dev/null &
    sleep 2
    # Ollama tag format: <name>:<version>[-<quant>]  e.g. gemma3:1b-q4_K_M
    # ollama pull handles the download from the Ollama registry.
    ollama pull "${MODEL}${QUANT:+-$QUANT}"
    ;;
  llamacpp)
    python3 -c "import llama_cpp" 2>/dev/null || install_llamacpp
    HF_REPO=$(_cfg hf_repo)
    HF_FILE=$(_cfg hf_file)
    [[ -z "$HF_REPO" ]] && { echo "ERROR: hf_repo not set in config"; exit 1; }
    download_gguf "$HF_REPO" "$HF_FILE" "$MODEL"
    python3 -m llama_cpp.server --model "$MODEL" --port 8080 --host 0.0.0.0 &
    ;;
  vllm)
    python3 -c "import vllm" 2>/dev/null || install_vllm
    # Pre-download so the server starts without waiting for a large model fetch.
    download_hf_model "$MODEL"
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" ${QUANT:+--quantization "$QUANT"} &
    ;;
  *)
    echo "Unknown engine: $ENGINE"; exit 1 ;;
esac

echo "$ENGINE inference server starting in background (PID $!)."
