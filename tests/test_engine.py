from pathlib import Path
from unittest.mock import MagicMock, patch
from src.finance_risk_rag.engine import RAGEngine, LLMClientWrapper

@patch('src.finance_risk_rag.engine.OpenAI')
def test_llm_client_call(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value.choices[0].message.content = "Test Answer"

    config = MagicMock()
    config.llm_api_key = "test_key"
    config.llm_base_url = "test_url"
    config.llm_model_name = "test_model"

    wrapper = LLMClientWrapper(config)
    ans = wrapper.call([{"role": "user", "content": "hello"}])
    assert ans == "Test Answer"

@patch('src.finance_risk_rag.engine.chromadb.PersistentClient')
def test_rag_engine_init(mock_chroma):
    config = MagicMock()
    config.chroma_db_dir = Path("/tmp/rag_db")
    config.llm_api_key = "test"
    config.llm_base_url = "https://api.example.com"
    config.llm_model_name = "test-model"

    engine = RAGEngine(config)
    assert engine.db_client is not None
    mock_chroma.assert_called_once()
