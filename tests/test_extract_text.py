"""
Finance-Risk-RAG 文档处理器单元测试
"""

from unittest.mock import MagicMock, patch
from finance_risk_rag.extract_text import DocumentProcessor
from finance_risk_rag.config import Config


def test_document_processor_initialization():
    config = Config()
    processor = DocumentProcessor(config=config)
    assert processor.config == config


@patch('finance_risk_rag.extract_text.OpenAI')
def test_classify_with_ai(mock_openai):
    # 配置 mock
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"type": "审计报告", "confidence": 0.95}'
    mock_client.chat.completions.create.return_value = mock_response

    config = Config()
    config.llm_api_key = "fake-key"
    processor = DocumentProcessor(config=config)

    result = processor.classify_with_ai("这是一些示例文本")
    assert result["type"] == "审计报告"
    assert result["confidence"] == 0.95
