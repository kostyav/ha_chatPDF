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

# ── Native mode ────────────────────────────────────────────────────────────────
case "$ENGINE" in
  ollama)
    ollama serve &>/dev/null &
    sleep 2
    # Append quantization tag when provided (e.g. gemma:3-1b-it:q4_0)
    ollama pull "${MODEL}${QUANT:+:$QUANT}"
    ;;
  llamacpp)
    # MODEL should be a local .gguf path; quantization is baked into that file.
    llama-server --model "$MODEL" --port 8080 --host 0.0.0.0 &
    ;;
  vllm)
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" ${QUANT:+--quantization "$QUANT"} &
    ;;
  *)
    echo "Unknown engine: $ENGINE"; exit 1 ;;
esac

echo "$ENGINE inference server starting in background (PID $!)."
