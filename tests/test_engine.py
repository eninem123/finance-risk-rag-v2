from unittest.mock import patch

from src.finance_risk_rag.engine import RAGEngine, TextChunker
from src.finance_risk_rag.models import ChunkConfig


def test_text_chunker():
    config = ChunkConfig(chunk_size=50, overlap=10)
    chunker = TextChunker(config)
    text = "这是一个很长的句子，我们希望它能被正确地切分成多个块。以便于向量检索。"
    chunks = chunker.chunk(text)
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk) <= config.chunk_size


@patch("src.finance_risk_rag.engine.RAGDatabase")
@patch("src.finance_risk_rag.engine.LLMClientWrapper")
def test_rag_engine_query(mock_llm, mock_db):
    mock_llm_instance = mock_llm.return_value
    mock_llm_instance.ask.return_value = "这是模拟的回答"

    mock_db_instance = mock_db.return_value
    mock_db_instance.query.return_value = [
        {"content": "相关片段", "metadata": {"source": "test.txt", "chunk_index": 0}}
    ]

    engine = RAGEngine()
    result = engine.query("测试问题")
    assert result.answer == "这是模拟的回答"
    assert len(result.sources) == 1
    assert result.sources[0]["source"] == "test.txt"
