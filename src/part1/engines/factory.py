"""Engine factory — returns an OpenAI-compatible client for the configured engine."""
from pathlib import Path
import yaml
from openai import OpenAI

DEFAULT_CONFIG = Path(__file__).parents[1] / "config" / "config.yaml"

# All three engines expose an OpenAI-compatible REST API; only base_url differs.
ENGINE_BASE_URLS = {
    "ollama":   "http://localhost:11434/v1",
    "llamacpp": "http://localhost:8080/v1",
    "vllm":     "http://localhost:8000/v1",
}


def load_config(path=None) -> dict:
    return yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())


def get_client(config_path=None) -> tuple[OpenAI, str]:
    """Return (OpenAI client, model_name) for the engine described in the config."""
    cfg = load_config(config_path)
    engine = cfg["engine"]
    if engine not in ENGINE_BASE_URLS:
        raise ValueError(f"Unknown engine '{engine}'. Choose from: {list(ENGINE_BASE_URLS)}")
    client = OpenAI(base_url=ENGINE_BASE_URLS[engine], api_key="none")
    return client, cfg["model"]
