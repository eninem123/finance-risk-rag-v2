from unittest.mock import patch

from src.finance_risk_rag.processor import DocumentProcessor


def test_document_processor_init():
    processor = DocumentProcessor()
    assert processor._llm_client is not None


@patch("src.finance_risk_rag.processor.LLMClientWrapper")
def test_classify_document(mock_llm):
    mock_instance = mock_llm.return_value
    mock_instance.is_available = True
    mock_instance.call.return_value = '{"type": "财报", "confidence": 0.95, "reason": "test"}'

    processor = DocumentProcessor()
    result = processor.classify_document("some text")
    assert result["type"] == "财报"
    assert result["confidence"] == 0.95


def test_optimize_image_for_ocr():
    from PIL import Image

    processor = DocumentProcessor()
    img = Image.new("RGB", (100, 100), color="white")
    optimized = processor.optimize_image_for_ocr(img)
    assert optimized.mode == "1"
