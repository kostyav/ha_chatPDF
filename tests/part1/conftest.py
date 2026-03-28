"""Shared fixtures and parametrize data for part1 tests."""
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

_HF_AUTH_ERRORS = (GatedRepoError, RepositoryNotFoundError)

CONFIG_DIR = Path(__file__).parents[2] / "src" / "part1" / "config"
REPO_ROOT  = Path(__file__).parents[2]

# Each entry: (config_file, engine, expected_port)
ENGINE_CONFIGS = [
    (CONFIG_DIR / "config.ollama.yaml",   "ollama",   11434),
    (CONFIG_DIR / "config.llamacpp.yaml", "llamacpp", 8080),
    (CONFIG_DIR / "config.vllm.yaml",     "vllm",     8000),
]

# Maps engine name → session fixture name (used by parametrised integration tests)
ENGINE_SERVER_FIXTURE = {
    "ollama":   "ollama_server",
    "llamacpp": "llamacpp_server",
    "vllm":     "vllm_server",
}


@pytest.fixture
def default_config_path():
    return CONFIG_DIR / "config.yaml"


# ── Server lifecycle helpers ───────────────────────────────────────────────────

def _port_open(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("localhost", port)) == 0


def _wait_for_port(port: int, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_open(port):
            return True
        time.sleep(1)
    return False


def _start_server(cmd: list[str], port: int, timeout: int):
    """Start cmd if port is not already bound. Returns (proc|None, started)."""
    if _port_open(port):
        return None, False          # already running externally — don't manage it
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not _wait_for_port(port, timeout):
        proc.terminate()
        proc.wait()
        pytest.skip(f"Server on port {port} did not become ready within {timeout}s")
    return proc, True


_HF_GATE_HINT = (
    "This is a gated HuggingFace model. To access it:\n"
    "  1. Accept the terms at https://huggingface.co/<model_id>\n"
    "  2. Generate a token at https://huggingface.co/settings/tokens\n"
    "  3. Run:  huggingface-cli login   or   export HF_TOKEN=<your_token>"
)


def _download_gguf(hf_repo: str, hf_file: str, dest: Path) -> None:
    """Download a single GGUF file from HuggingFace into dest (skipped if exists)."""
    if dest.exists():
        return
    print(f"\nDownloading {hf_repo}/{hf_file} → {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        src = hf_hub_download(repo_id=hf_repo, filename=hf_file)
    except _HF_AUTH_ERRORS:
        pytest.skip(_HF_GATE_HINT.replace("<model_id>", hf_repo))
    shutil.copy(src, dest)


def _download_hf_model(model_id: str) -> None:
    """Pre-populate the HF cache for model_id so the server starts without fetching."""
    print(f"\nPre-downloading HuggingFace model: {model_id} ...")
    try:
        snapshot_download(model_id)
    except _HF_AUTH_ERRORS:
        pytest.skip(_HF_GATE_HINT.replace("<model_id>", model_id))


# ── Per-engine session fixtures ───────────────────────────────────────────────

@pytest.fixture(scope="session")
def ollama_server():
    """Ensure the Ollama inference server is running on :11434 and the model is pulled."""
    cfg = yaml.safe_load((CONFIG_DIR / "config.ollama.yaml").read_text())
    # Ollama encodes quantization inside its own tag (e.g. gemma3:1b already uses Q4_K_M).
    # Appending quantization as a suffix (gemma3:1b-q4_K_M) produces an invalid tag.
    model_tag = cfg["model"]

    proc, started = _start_server(["ollama", "serve"], port=11434, timeout=30)
    subprocess.run(["ollama", "pull", model_tag], check=True)
    yield
    if started:
        proc.terminate()
        proc.wait()


@pytest.fixture(scope="session")
def llamacpp_server():
    """Download the GGUF from HuggingFace (if needed) then serve it on :8080."""
    cfg  = yaml.safe_load((CONFIG_DIR / "config.llamacpp.yaml").read_text())
    dest = REPO_ROOT / cfg["model"]

    _download_gguf(cfg["hf_repo"], cfg["hf_file"], dest)

    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", str(dest), "--port", "8080", "--host", "0.0.0.0",
    ]
    proc, started = _start_server(cmd, port=8080, timeout=60)
    yield
    if started:
        proc.terminate()
        proc.wait()


@pytest.fixture(scope="session")
def vllm_server():
    """Download the HuggingFace model (if needed) then serve it on :8000."""
    cfg = yaml.safe_load((CONFIG_DIR / "config.vllm.yaml").read_text())

    _download_hf_model(cfg["model"])

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg["model"],
    ]
    if cfg.get("quantization"):
        cmd += ["--quantization", cfg["quantization"]]
    proc, started = _start_server(cmd, port=8000, timeout=300)
    yield
    if started:
        proc.terminate()
        proc.wait()
