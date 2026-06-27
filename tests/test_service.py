"""
RiskAnalysisService 单元测试
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.finance_risk_rag.models import Entity, ExtractionResult
from src.finance_risk_rag.service import RiskAnalysisService


class TestRiskAnalysisService(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock()
        # Ensure config has necessary attributes for RiskAnalysisService
        self.mock_config.llm_api_key = None
        self.mock_config.llm_base_url = "http://localhost"
        self.mock_config.llm_model_name = "test-model"

        self.mock_processor = MagicMock()
        self.mock_pipeline = MagicMock()
        self.mock_engine = MagicMock()
        self.mock_llm = MagicMock()
        self.mock_llm.is_available = False  # Disable AI summary by default in tests

        self.service = RiskAnalysisService(
            config=self.mock_config,
            processor=self.mock_processor,
            pipeline=self.mock_pipeline,
            engine=self.mock_engine,
            llm_client=self.mock_llm,
        )

    def test_run_full_analysis_file(self):
        # 准备 Mock 返回值
        self.mock_processor.process_single_pdf.return_value = {
            "name": "test.pdf",
            "text": "sample text",
            "classification": {"type": "审计报告", "confidence": 0.9, "reason": "test"},
            "hash": "abc",
        }

        mock_entities = [Entity(type="RISK", text="debt", risk_score=30, confidence=0.8)]
        self.mock_pipeline.process.return_value = ExtractionResult(
            entities=mock_entities, total_risk_score=30, risk_level="低风险"
        )
        self.mock_llm.chat.return_value = "Test Summary"
        self.mock_llm.is_available = True

        # 执行
        # We need to mock is_file and is_dir or use a real path
        mock_path = MagicMock(spec=Path)
        mock_path.is_file.return_value = True
        mock_path.suffix.lower.return_value = ".pdf"
        mock_path.name = "test.pdf"
        mock_path.__str__.return_value = "test.pdf"
        mock_path.with_suffix.return_value = Path("test.txt")

        result = self.service.run_full_analysis(mock_path)

        # 断言
        self.assertEqual(result["document_info"]["name"], "test.pdf")
        self.assertEqual(result["classification"]["type"], "审计报告")
        self.assertEqual(result["risk_analysis"]["total_risk_score"], 30)
        self.mock_processor.process_single_pdf.assert_called_once()
        self.mock_pipeline.process.assert_called_once()

    def test_generate_report(self):
        # 准备数据
        analysis_data = {
            "document_info": {
                "name": "test.pdf",
                "analyzed_at": "2024-01-01",
            },
            "classification": {"type": "审计报告", "confidence": 0.9, "reason": "test"},
            "risk_analysis": {
                "risk_level": "中风险",
                "total_risk_score": 45,
                "total_entities": 1,
                "entities": [
                    {
                        "type": "RISK",
                        "text": "bad debt",
                        "risk_score": 45,
                        "confidence": 0.85,
                        "source": "bert",
                    }
                ],
            },
            "executive_summary": "Test AI Summary",
        }

        # 执行
        report = self.service.generate_report(analysis_data)

        # 断言
        self.assertIn("# 财务风险分析报告: test.pdf", report)
        self.assertIn("中风险", report)
        self.assertIn("bad debt", report)
        self.assertIn("Test AI Summary", report)
        self.assertIn("### 🟡 关注类建议", report)


if __name__ == "__main__":
    unittest.main()
