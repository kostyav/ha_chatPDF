"""Shared fixtures and parametrize data for part2 tests."""
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest
import yaml
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

CONFIG_DIR  = Path(__file__).parents[2] / "src" / "part2" / "config"
REPO_ROOT   = Path(__file__).parents[2]
EXAMPLE_DIR = Path(__file__).parents[2] / "src" / "part2" / "example_data"

# (config_file, engine, port)
ENGINE_CONFIGS = [
    (CONFIG_DIR / "config.ollama.yaml",   "ollama",   11434),
    (CONFIG_DIR / "config.llamacpp.yaml", "llamacpp", 8080),
    (CONFIG_DIR / "config.vllm.yaml",     "vllm",     8000),
]

ENGINE_SERVER_FIXTURE = {
    "ollama":   "ollama_server",
    "llamacpp": "llamacpp_server",
    "vllm":     "vllm_server",
}

_HF_AUTH_ERRORS = (GatedRepoError, RepositoryNotFoundError)
_HF_GATE_HINT = (
    "Gated HuggingFace model. Accept terms, then:\n"
    "  huggingface-cli login   or   export HF_TOKEN=<token>"
)


@pytest.fixture
def default_config_path():
    return CONFIG_DIR / "config.yaml"


# ── Port / HTTP helpers ────────────────────────────────────────────────────────

def _wait_for_http(port: int, timeout: int) -> bool:
    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _start_server(cmd: list[str], port: int, timeout: int):
    if _wait_for_http(port, timeout=5):
        return None, False
    stderr_file = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr_file)
    if not _wait_for_http(port, timeout):
        stderr_file.seek(0)
        last = stderr_file.read().decode(errors="replace")[-1000:]
        proc.terminate(); proc.wait()
        pytest.skip(f"Server on :{port} not ready within {timeout}s.\n{last}")
    return proc, True


# ── Server fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def ollama_server():
    cfg = yaml.safe_load((CONFIG_DIR / "config.ollama.yaml").read_text())
    proc, started = _start_server(["ollama", "serve"], port=11434, timeout=30)
    subprocess.run(["ollama", "pull", cfg["model"]], check=True)
    yield
    if started:
        proc.terminate(); proc.wait()


@pytest.fixture(scope="session")
def llamacpp_server():
    cfg  = yaml.safe_load((CONFIG_DIR / "config.llamacpp.yaml").read_text())
    dest = REPO_ROOT / cfg["model"]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            src = hf_hub_download(repo_id=cfg["hf_repo"], filename=cfg["hf_file"])
        except _HF_AUTH_ERRORS:
            pytest.skip(_HF_GATE_HINT)
        shutil.copy(src, dest)
    cmd = [sys.executable, "-m", "llama_cpp.server", "--model", str(dest), "--port", "8080"]
    proc, started = _start_server(cmd, port=8080, timeout=60)
    yield
    if started:
        proc.terminate(); proc.wait()


@pytest.fixture(scope="session")
def vllm_server():
    cfg = yaml.safe_load((CONFIG_DIR / "config.vllm.yaml").read_text())
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg["model"], "--enforce-eager", "--max-model-len", "1024",
    ]
    if cfg.get("quantization"):
        cmd += ["--quantization", cfg["quantization"]]
    proc, started = _start_server(cmd, port=8000, timeout=300)
    yield
    if started:
        proc.terminate(); proc.wait()
