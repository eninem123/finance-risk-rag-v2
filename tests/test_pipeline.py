from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.finance_risk_rag.config import Config
from src.finance_risk_rag.engine import RAGEngine
from src.finance_risk_rag.extractor import EntityExtractionPipeline
from src.finance_risk_rag.processor import DocumentProcessor


def test_full_pipeline_integration(tmp_path):
    # Setup mock config
    conf = Config()
    conf.base_dir = tmp_path
    conf.docs_dir = tmp_path / "docs"
    conf.cache_dir = tmp_path / "cache"
    conf.chroma_db_dir = tmp_path / "db"
    conf.knowledge_base_dir = tmp_path / "kb"
    for d in [conf.docs_dir, conf.cache_dir, conf.chroma_db_dir, conf.knowledge_base_dir]:
        d.mkdir(parents=True)

    # Create a dummy risk entity file
    import json

    risk_rules = {"market_risk": {"keywords": ["跌幅", "波动"], "risk_score": 15}}
    (conf.knowledge_base_dir / "risk_entities.json").write_text(json.dumps(risk_rules))

    # Create a dummy PDF
    pdf_path = conf.docs_dir / "report.pdf"
    pdf_path.write_text("Dummy PDF content")

    # Mock LLM and OCR
    mock_llm = MagicMock()
    mock_llm.is_available = True
    mock_llm.chat.return_value = '{"type": "财报", "confidence": 0.95, "reason": "test"}'
    mock_llm.ask.return_value = "The market risk is moderate."

    with patch("pdfplumber.open"), patch(
        "pytesseract.image_to_string", return_value="近期市场波动巨大，跌幅明显。"
    ), patch("chromadb.PersistentClient"), patch(
        "chromadb.utils.embedding_functions.ONNXMiniLM_L6_V2"
    ):

        # 1. Process
        processor = DocumentProcessor(config=conf, llm_client=mock_llm)
        # Manually mock extract_text_from_pdf to bypass pdfplumber/tesseract
        processor.extract_text_from_pdf = MagicMock(
            return_value=("近期市场波动巨大，跌幅明显。", 1)
        )
        processor.process_directory(max_workers=1)

        extracted_txt = conf.docs_dir / "report.txt"
        assert extracted_txt.exists()

        # 2. Extract
        extractor = EntityExtractionPipeline(config=conf)
        result = extractor.process(extracted_txt)
        assert len(result.entities) > 0
        assert any(e.text == "波动" for e in result.entities)

        # 3. Query
        engine = RAGEngine(config=conf, llm_client=mock_llm)
        # Mock collection for engine
        engine._collection = MagicMock()
        engine._collection.query.return_value = {
            "documents": [["context"]],
            "metadatas": [[{"source": "report.pdf"}]],
        }

        query_res = engine.query("市场风险如何？")
        assert "moderate" in query_res.answer
