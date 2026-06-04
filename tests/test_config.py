import os
from pathlib import Path
from src.finance_risk_rag.config import Config, get_config

def test_config_defaults():
    config = Config()
    assert config.llm_provider == "moonshot"
    assert config.chunk_size == 800
    assert isinstance(config.base_dir, Path)

def test_config_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    config = Config()
    assert config.llm_provider == "openai"
    assert config.chunk_size == 500

def test_get_config_singleton():
    c1 = get_config()
    c2 = get_config()
    assert c1 is c2

def test_config_paths():
    config = Config()
    assert config.docs_dir.exists() or not config.docs_dir.is_absolute()
    # base_dir is absolute due to resolve() in dataclass
    assert config.base_dir.is_absolute()
