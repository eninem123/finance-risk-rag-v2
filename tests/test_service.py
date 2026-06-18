import json
from unittest.mock import patch

from src.finance_risk_rag.config import Config
from src.finance_risk_rag.service import RiskAnalysisService


def test_risk_analysis_service_orchestration(tmp_path):
    # Setup mock config
    conf = Config()
    conf.base_dir = tmp_path
    conf.docs_dir = tmp_path / "docs"
    conf.cache_dir = tmp_path / "cache"
    conf.chroma_db_dir = tmp_path / "db"
    conf.knowledge_base_dir = tmp_path / "kb"
    for d in [conf.docs_dir, conf.cache_dir, conf.chroma_db_dir, conf.knowledge_base_dir]:
        d.mkdir(parents=True)

    # Dummy rules
    risk_rules = {"test_risk": {"keywords": ["risk"], "risk_score": 10}}
    (conf.knowledge_base_dir / "risk_entities.json").write_text(json.dumps(risk_rules))

    # Mock PDF
    pdf_path = conf.docs_dir / "test.pdf"
    pdf_path.write_text("dummy")
    # Mock corresponding TXT so it exists for RAG
    txt_path = pdf_path.with_suffix(".txt")
    txt_path.write_text("dummy text")

    with patch("src.finance_risk_rag.service.DocumentProcessor") as mock_proc_cls, patch(
        "src.finance_risk_rag.service.EntityExtractionPipeline"
    ) as mock_pipe_cls, patch("src.finance_risk_rag.service.RAGEngine") as mock_engine_cls:
        mock_proc = mock_proc_cls.return_value
        mock_proc.process_single_pdf.return_value = {
            "text": "There is a major risk here.",
            "classification": {"type": "Report", "confidence": 0.9},
        }

        mock_pipe = mock_pipe_cls.return_value
        from src.finance_risk_rag.models import Entity, ExtractionResult

        mock_pipe.process.return_value = ExtractionResult(
            entities=[
                Entity(
                    type="test_risk",
                    text="risk",
                    risk_score=10,
                    confidence=1.0,
                    start_char=17,
                    end_char=21,
                )
            ],
            total_risk_score=10,
            risk_level="Low",
        )

        service = RiskAnalysisService(conf)
        report = service.analyze_document(pdf_path)

        assert report["document_name"] == "test.pdf"
        assert report["classification"]["type"] == "Report"
        assert report["risk_analysis"]["total_risk_score"] == 10
        assert "发现 1 个风险实体" in report["summary"]

        # Verify engine was called to add doc
        mock_engine = mock_engine_cls.return_value
        mock_engine.add_documents.assert_called_once()
