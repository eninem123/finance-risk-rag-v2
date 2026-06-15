from unittest.mock import patch

import pytest

from src.finance_risk_rag.config import Config
from src.finance_risk_rag.models import Entity, ExtractionResult, QueryResult
from src.finance_risk_rag.service import RiskAnalysisService


@pytest.fixture
def mock_config(tmp_path):
    conf = Config()
    conf.docs_dir = tmp_path / "docs"
    conf.cache_dir = tmp_path / "cache"
    conf.docs_dir.mkdir()
    conf.cache_dir.mkdir()
    return conf


def test_analyze_document(mock_config):
    service = RiskAnalysisService(config=mock_config)
    pdf_path = mock_config.docs_dir / "test.pdf"
    pdf_path.write_text("dummy content")

    mock_proc_res = {
        "text": "sample text",
        "classification": {"type": "Audit Report", "confidence": 0.9},
        "ocr_pages": 1,
        "hash": "abc",
    }

    mock_ext_res = ExtractionResult(
        entities=[Entity(type="RISK", text="debt", risk_score=50, confidence=1.0)],
        total_risk_score=50,
        risk_level="Medium",
    )

    mock_query_res = QueryResult(answer="Analysis answer", sources=[])

    with patch(
        "src.finance_risk_rag.service.DocumentProcessor.process_single_pdf",
        return_value=mock_proc_res,
    ), patch(
        "src.finance_risk_rag.service.EntityExtractionPipeline.process", return_value=mock_ext_res
    ), patch(
        "src.finance_risk_rag.service.RAGEngine.query", return_value=mock_query_res
    ):

        report = service.analyze_document(pdf_path)

        assert report["document_name"] == "test.pdf"
        assert report["risk_assessment"]["level"] == "Medium"
        assert len(report["risk_assessment"]["entities"]) == 1
        assert report["ai_analysis"]["summary"] == "Analysis answer"


def test_generate_report_markdown(mock_config):
    service = RiskAnalysisService(config=mock_config)
    report = {
        "document_name": "test.pdf",
        "classification": {"type": "Audit Report", "confidence": 0.9},
        "risk_assessment": {
            "level": "Medium",
            "score": 50,
            "entities": [
                {
                    "type": "RISK",
                    "text": "debt",
                    "risk_score": 50,
                    "source": "rule",
                    "context": "context",
                }
            ],
        },
        "ai_analysis": {"summary": "summary"},
        "metadata": {"ocr_pages": 1},
    }

    md = service.generate_report_markdown(report)
    assert "# 财务风险分析报告: test.pdf" in md
    assert "Medium" in md
    assert "RISK" in md
