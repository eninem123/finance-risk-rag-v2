from unittest.mock import MagicMock, patch

from src.finance_risk_rag.config import Config
from src.finance_risk_rag.engine import RAGEngine
from src.finance_risk_rag.extractor import EntityExtractionPipeline
from src.finance_risk_rag.processor import DocumentProcessor



    # Create a dummy risk entity file
    import json

    risk_rules = {"market_risk": {"keywords": ["跌幅", "波动"], "risk_score": 15}}
    (conf.knowledge_base_dir / "risk_entities.json").write_text(json.dumps(risk_rules))

    def test_merge_and_arbitrate_overlap(self):
        # 准备两个重叠的实体
        e1 = Entity(
            type="RISK",
            text="财务风险",
            risk_score=30,
            confidence=1.0,
            metadata={"start_char": 0, "end_char": 4},
        )
        e2 = Entity(
            type="RISK",
            text="风险",
            risk_score=10,
            confidence=0.9,
            metadata={"start_char": 2, "end_char": 4},
        )

        merged = self.pipeline._merge_and_arbitrate([e1], [e2])

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

        result = self.pipeline.process("文本样本")

        self.assertEqual(result.total_risk_score, 0)
        self.assertEqual(result.risk_level, "低风险")

        # 3. Query
        engine = RAGEngine(config=conf, llm_client=mock_llm)
        # Mock collection for engine
        engine._collection = MagicMock()
        engine._collection.query.return_value = {
            "documents": [["context"]],
            "metadatas": [[{"source": "report.pdf"}]],
        }

if __name__ == "__main__":
    unittest.main()
