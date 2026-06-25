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
        self.mock_processor = MagicMock()
        self.mock_extractor = MagicMock()
        self.service = RiskAnalysisService(
            config=self.mock_config, processor=self.mock_processor, extractor=self.mock_extractor
        )

    def test_run_full_analysis(self):
        # 准备 Mock 返回值
        mock_pdf = MagicMock(spec=Path)
        mock_pdf.is_file.return_value = True
        mock_pdf.suffix.lower.return_value = ".pdf"
        mock_pdf.name = "test.pdf"
        mock_pdf.with_suffix.return_value = MagicMock(spec=Path)
        mock_pdf.with_suffix.return_value.exists.return_value = False

        self.mock_processor.process_single_pdf.return_value = {
            "text": "sample text",
            "classification": {"type": "审计报告", "confidence": 0.9, "reason": "test"},
            "hash": "abc",
        }

        mock_entities = [Entity(type="RISK", text="debt", risk_score=30, confidence=0.8)]
        self.mock_extractor.process.return_value = ExtractionResult(
            entities=mock_entities, total_risk_score=30, risk_level="低风险"
        )

        # 执行
        result = self.service.run_full_analysis(mock_pdf)

        # 断言
        self.assertEqual(result["document_info"]["name"], "test.pdf")
        self.assertEqual(result["classification"]["type"], "审计报告")
        self.assertEqual(result["risk_analysis"]["total_risk_score"], 30)
        self.mock_processor.process_single_pdf.assert_called_once_with(mock_pdf)
        self.mock_extractor.process.assert_called_once()

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
        }

        # 执行
        report = self.service.generate_report(analysis_data)

        # 断言
        self.assertIn("# 财务风险分析报告: test.pdf", report)
        self.assertIn("中风险", report)
        self.assertIn("bad debt", report)
        self.assertIn("💡 **建议**", report)


if __name__ == "__main__":
    unittest.main()
