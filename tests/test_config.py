from pathlib import Path
from src.finance_risk_rag.config import get_config

def test_config_defaults():
    config = get_config()
    assert config.llm_provider == "moonshot"
    assert config.llm_model_name == "moonshot-v1-8k"
    assert isinstance(config.base_dir, Path)
    assert config.base_dir.exists()

def test_config_paths():
    config = get_config()
    assert config.docs_dir.is_absolute()
    assert config.knowledge_base_dir.is_absolute()
    assert config.chroma_db_dir.is_absolute()
