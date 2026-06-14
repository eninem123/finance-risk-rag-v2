import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from finance_risk_rag.service import RiskAnalysisService
from finance_risk_rag.config import Config
from finance_risk_rag.models import ExtractionResult, Entity

@pytest.fixture
def mock_config(tmp_path):
    config = Config()
    config.docs_dir = tmp_path / "docs"
    config.docs_dir.mkdir()
    return config

def test_service_initialization(mock_config):
    service = RiskAnalysisService(mock_config)
    assert service.processor is not None
    assert service.extractor is not None
    assert service.rag_engine is not None

@patch("finance_risk_rag.processor.DocumentProcessor.process_single_pdf")
@patch("finance_risk_rag.extractor.EntityExtractionPipeline.process")
def test_analyze_document(mock_extract, mock_process, mock_config):
    service = RiskAnalysisService(mock_config)
    pdf_path = mock_config.docs_dir / "test.pdf"

    mock_process.return_value = {
        "text": "sample text",
        "classification": {"type": "审计报告", "confidence": 0.9}
    }

    mock_extract.return_value = ExtractionResult(
        entities=[Entity(type="RISK", text="danger", risk_score=50, confidence=1.0)],
        total_risk_score=50,
        risk_level="中风险"
    )

    report = service.analyze_document(pdf_path)

    assert report["document"]["name"] == "test.pdf"
    assert report["document"]["type"] == "审计报告"
    assert report["risk_analysis"]["total_risk_score"] == 50
    assert "发现 1 处潜在风险点" in report["summary"]
