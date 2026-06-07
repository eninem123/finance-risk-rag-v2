from unittest.mock import patch

from finance_risk_rag.engine import RAGEngine, TextChunker
from finance_risk_rag.models import ChunkConfig


def test_text_chunker():
    config = ChunkConfig(chunk_size=50, overlap=10)
    chunker = TextChunker(config)
    text = "这是一个很长的句子，用来测试分块功能。我们希望它能正确地按照配置进行切分。"
    chunks = chunker.chunk(text)
    assert len(chunks) > 0
    assert all(len(c) <= 50 for c in chunks)


@patch("finance_risk_rag.engine.RAGDatabase")
@patch("finance_risk_rag.engine.LLMClientWrapper")
def test_rag_engine_query(mock_llm, mock_db):
    mock_db_instance = mock_db.return_value
    mock_db_instance.query.return_value = [
        {
            "content": "测试内容",
            "metadata": {"source": "test.txt", "chunk_index": 0},
            "distance": 0.1,
        }
    ]

    mock_llm_instance = mock_llm.return_value
    mock_llm_instance.ask.return_value = "这是模拟回答"

    engine = RAGEngine()
    result = engine.query("测试问题")

    assert result.answer == "这是模拟回答"
    assert len(result.sources) == 1
    assert result.sources[0]["source"] == "test.txt"
