import os
from finance_risk_rag.config import Config

def test_config_env_override():
    os.environ["LLM_MODEL_NAME"] = "test-model"
    os.environ["CHUNK_SIZE"] = "500"

    config = Config()
    assert config.llm_model_name == "test-model"
    assert config.chunk_size == 500

    # Cleanup
    del os.environ["LLM_MODEL_NAME"]
    del os.environ["CHUNK_SIZE"]

def test_config_paths(tmp_path):
    config = Config(base_dir=tmp_path)
    assert config.docs_dir == tmp_path / "docs"
    assert config.log_dir == tmp_path / "logs"
