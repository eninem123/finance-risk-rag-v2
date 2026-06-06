import pytest
from src.finance_risk_rag.config import get_config

def test_config():
    config = get_config()
    assert config.llm_provider == "moonshot"
    assert config.docs_dir.name == "docs"
