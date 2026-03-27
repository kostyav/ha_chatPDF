"""Shared fixtures and parametrize data for part1 tests."""
import pytest
from pathlib import Path

CONFIG_DIR = Path(__file__).parents[2] / "src" / "part1" / "config"

# Each entry: (config_file, engine, expected_port)
ENGINE_CONFIGS = [
    (CONFIG_DIR / "config.ollama.yaml",   "ollama",   11434),
    (CONFIG_DIR / "config.llamacpp.yaml", "llamacpp", 8080),
    (CONFIG_DIR / "config.vllm.yaml",     "vllm",     8000),
]


@pytest.fixture
def default_config_path():
    return CONFIG_DIR / "config.yaml"
