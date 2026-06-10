"""
Finance-Risk-RAG 配置模块单元测试
"""

from finance_risk_rag.config import Config, get_config


def test_config_initialization():
    config = Config()
    assert config.base_dir.exists()
    assert config.llm_provider == "moonshot"
    assert isinstance(config.chunk_size, int)


def test_get_config_singleton():
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    config = Config()
    assert config.llm_api_key == "test-key"
    assert config.llm_provider == "openai"


def test_config_paths():
    config = Config()
    assert config.docs_dir.name == "docs"
    assert config.knowledge_base_dir.name == "knowledge_base"
    assert config.risk_entities_path.name == "risk_entities.json"
