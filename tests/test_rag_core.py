from unittest.mock import MagicMock, patch
from finance_risk_rag.rag_core import TextChunker, LLMClientWrapper
from finance_risk_rag.models import ChunkConfig

def test_text_chunker():
    config = ChunkConfig(chunk_size=50, overlap=10)
    chunker = TextChunker(config)
    text = "这是一个非常长的句子，用于测试分块逻辑是否正常工作。" * 5
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 50

@patch("openai.OpenAI")
def test_llm_client_wrapper(mock_openai):
    mock_instance = mock_openai.return_value
    mock_instance.chat.completions.create.return_value.choices[0].message.content = "测试回答"

    # 设置 API KEY 避开初始化警告
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        from finance_risk_rag.config import Config
        config = Config()
        client = LLMClientWrapper(config)
        answer = client.ask("问题", "上下文")
        assert answer == "测试回答"
