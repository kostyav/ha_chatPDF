"""Tests for config loading and engine factory across all configurations."""
import pytest
from openai import OpenAI
from pathlib import Path

from src.part2.engines.factory import load_config, get_client, ENGINE_BASE_URLS
from .conftest import ENGINE_CONFIGS


class TestLoadConfig:
    def test_default_has_required_keys(self, default_config_path):
        cfg = load_config(default_config_path)
        for key in ("engine", "model", "embedding_model"):
            assert key in cfg

    def test_default_engine_is_valid(self, default_config_path):
        assert load_config(default_config_path)["engine"] in ENGINE_BASE_URLS

    @pytest.mark.parametrize("config_path,engine,_port", ENGINE_CONFIGS)
    def test_engine_matches_file(self, config_path, engine, _port):
        assert load_config(config_path)["engine"] == engine

    @pytest.mark.parametrize("config_path,engine,_port", ENGINE_CONFIGS)
    def test_model_non_empty(self, config_path, engine, _port):
        assert load_config(config_path)["model"]

    @pytest.mark.parametrize("config_path,engine,_port", ENGINE_CONFIGS)
    def test_quantization_key_present(self, config_path, engine, _port):
        cfg = load_config(config_path)
        assert "quantization" in cfg  # may be empty string

    @pytest.mark.parametrize("config_path,engine,_port", ENGINE_CONFIGS)
    def test_retriever_section(self, config_path, engine, _port):
        cfg = load_config(config_path)
        assert "retriever" in cfg
        assert cfg["retriever"]["top_k"] > 0
        assert 0 < cfg["retriever"]["similarity_threshold"] < 1

    def test_unknown_engine_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("engine: unknown\nmodel: foo\n")
        with pytest.raises(ValueError, match="Unknown engine"):
            get_client(bad)


class TestGetClient:
    @pytest.mark.parametrize("config_path,engine,port", ENGINE_CONFIGS)
    def test_returns_openai_client(self, config_path, engine, port):
        client, model = get_client(config_path)
        assert isinstance(client, OpenAI)
        assert str(port) in str(client.base_url)

    @pytest.mark.parametrize("config_path,engine,_port", ENGINE_CONFIGS)
    def test_model_non_empty(self, config_path, engine, _port):
        _, model = get_client(config_path)
        assert model
