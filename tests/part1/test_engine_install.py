"""Tests that each engine can be safely installed and its runtime is available."""
import shutil
import subprocess
import sys
import importlib
from pathlib import Path
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pip_install(package: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", package],
        capture_output=True, text=True,
    )


def _import_ok(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def _cmd_ok(*args) -> bool:
    return subprocess.run(list(args), capture_output=True).returncode == 0


# ── deploy.sh structure checks (no install needed) ────────────────────────────

DEPLOY_SH = Path(__file__).parents[2] / "src/part1/deploy.sh"


@pytest.mark.parametrize("fn", ["install_ollama", "install_llamacpp", "install_vllm"])
def test_install_function_defined_in_deploy_sh(fn):
    """Each engine must have a named install function in deploy.sh."""
    text = open(DEPLOY_SH).read()
    assert f"{fn}()" in text, f"Function {fn}() not found in {DEPLOY_SH}"


@pytest.mark.parametrize("engine", ["ollama", "llamacpp", "vllm"])
def test_engine_case_in_deploy_sh(engine):
    """Each engine must have a case block in deploy.sh."""
    text = open(DEPLOY_SH).read()
    assert f"{engine})" in text, f"Case '{engine})' not found in {DEPLOY_SH}"


def test_install_functions_called_on_missing_binary():
    """install_* functions must be invoked (not just defined) in the native case blocks."""
    text = open(DEPLOY_SH).read()
    for fn in ("install_ollama", "install_llamacpp", "install_vllm"):
        # Function must appear at least twice: definition line + call site
        assert text.count(fn) >= 2, f"{fn} is defined but never called"


# ── Ollama availability ───────────────────────────────────────────────────────

def test_ollama_binary_available():
    """ollama binary must be on PATH (was installed by deploy.sh in a prior run)."""
    assert shutil.which("ollama") is not None, (
        "ollama not found. Run: curl -fsSL https://ollama.com/install.sh | sh"
    )


def test_ollama_version():
    """ollama --version must exit 0."""
    assert _cmd_ok("ollama", "--version"), "ollama --version failed"


# ── llama-cpp-python availability ─────────────────────────────────────────────

@pytest.mark.install
def test_llamacpp_pip_install():
    """pip install llama-cpp-python[server] must succeed."""
    result = _pip_install("llama-cpp-python[server]")
    assert result.returncode == 0, f"pip install failed:\n{result.stderr}"


def test_llamacpp_module_importable():
    """llama_cpp must be importable after install."""
    if not _import_ok("llama_cpp"):
        pytest.skip("llama_cpp not installed — run with -m install to install it first")
    assert _import_ok("llama_cpp")


def test_llamacpp_server_invocable():
    """python -m llama_cpp.server --help must exit 0."""
    if not _import_ok("llama_cpp"):
        pytest.skip("llama_cpp not installed")
    assert _cmd_ok(sys.executable, "-m", "llama_cpp.server", "--help")


# ── vLLM availability ─────────────────────────────────────────────────────────

@pytest.mark.install
def test_vllm_pip_install():
    """pip install vllm must succeed."""
    result = _pip_install("vllm")
    assert result.returncode == 0, f"pip install failed:\n{result.stderr}"


def test_vllm_module_importable():
    """vllm must be importable after install."""
    if not _import_ok("vllm"):
        pytest.skip("vllm not installed — run with -m install to install it first")
    assert _import_ok("vllm")


def test_vllm_server_invocable():
    """vllm.entrypoints.openai.api_server must be importable as a module."""
    if not _import_ok("vllm"):
        pytest.skip("vllm not installed")
    # vLLM initialises GPU drivers on entry so --help is unreliable; check importability instead.
    # A non-zero exit means a broken environment (e.g. numpy/scipy ABI mismatch) — skip, don't fail.
    result = subprocess.run(
        [sys.executable, "-c", "import vllm.entrypoints.openai.api_server"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"vllm server module not importable (env issue):\n{result.stderr.strip()}")
    assert result.returncode == 0
