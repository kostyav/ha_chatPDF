"""Test factory behaviour across all engine/model/quantization configurations."""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from openai import OpenAI

from src.part1.engines.factory import load_config, get_client, ENGINE_BASE_URLS
from .conftest import ENGINE_CONFIGS


# ── Unit tests (no live server required) ──────────────────────────────────────

class TestLoadConfig:
    def test_loads_engine(self, default_config_path):
        cfg = load_config(default_config_path)
        assert cfg["engine"] in ENGINE_BASE_URLS

    def test_loads_model(self, default_config_path):
        cfg = load_config(default_config_path)
        assert cfg["model"]

    @pytest.mark.parametrize("config_path,engine,_port", ENGINE_CONFIGS)
    def test_engine_in_each_config(self, config_path, engine, _port):
        cfg = load_config(config_path)
        assert cfg["engine"] == engine


class TestGetClient:
    @pytest.mark.parametrize("config_path,engine,port", ENGINE_CONFIGS)
    def test_client_base_url(self, config_path, engine, port):
        client, model = get_client(config_path)
        assert isinstance(client, OpenAI)
        assert str(port) in client.base_url.host or str(port) in str(client.base_url)

    def test_unknown_engine_raises(self, tmp_path):
        bad_cfg = tmp_path / "bad.yaml"
        bad_cfg.write_text("engine: unknown\nmodel: foo\n")
        with pytest.raises(ValueError, match="Unknown engine"):
            get_client(bad_cfg)

    @pytest.mark.parametrize("config_path,engine,_port", ENGINE_CONFIGS)
    def test_model_returned(self, config_path, engine, _port):
        _, model = get_client(config_path)
        assert model  # non-empty


# ── Integration tests (require a running inference server) ────────────────────

@pytest.mark.integration
@pytest.mark.parametrize("config_path,engine,_port", ENGINE_CONFIGS)
def test_chat_completion_per_engine(config_path, engine, _port):
    """Verify each engine returns a non-empty chat completion (needs live server)."""
    client, model = get_client(config_path)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with one word: OK"}],
        max_tokens=16,
    )
    assert resp.choices[0].message.content.strip()


@pytest.mark.integration
@pytest.mark.parametrize("config_path,engine,_port", ENGINE_CONFIGS)
def test_model_list_per_engine(config_path, engine, _port):
    """Verify the engine's /v1/models endpoint lists at least one model."""
    client, _ = get_client(config_path)
    models = client.models.list()
    assert len(list(models)) >= 1
