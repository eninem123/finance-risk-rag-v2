"""
RiskAnalysisService 单元测试
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock

from src.finance_risk_rag.models import Entity, ExtractionResult
from src.finance_risk_rag.service import RiskAnalysisService
from src.finance_risk_rag.exceptions import OCRError


class TestRiskAnalysisService(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # 确保 mock_config 的 llm_base_url 是字符串
        self.mock_config = MagicMock()
        self.mock_config.llm_base_url = "http://localhost"
        self.mock_config.llm_api_key = "test_key"
        self.mock_config.llm_model_name = "test_model"
        self.mock_config.chroma_db_dir = Path(self.test_dir) / "test_db"

        self.mock_processor = MagicMock()
        self.mock_extractor = MagicMock()
        self.mock_engine = MagicMock()

        self.service = RiskAnalysisService(
            config=self.mock_config,
            processor=self.mock_processor,
            extractor=self.mock_extractor,
            engine=self.mock_engine
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_run_full_analysis_success(self):
        # 准备 Mock 返回值
        pdf_path = Path(self.test_dir) / "test.pdf"
        pdf_path.touch()

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
        result = self.service.run_full_analysis(pdf_path)

        # 断言
        self.assertEqual(result["document_info"]["name"], "test.pdf")
        self.assertEqual(result["classification"]["type"], "审计报告")
        self.assertEqual(result["risk_analysis"]["total_risk_score"], 30)
        self.mock_processor.process_single_pdf.assert_called_once_with(pdf_path)
        self.mock_extractor.process.assert_called_once()

    def test_run_full_analysis_error_handling(self):
        # 测试 OCR 错误处理
        pdf_path = Path(self.test_dir) / "error.pdf"
        pdf_path.touch()

        self.mock_processor.process_single_pdf.side_effect = OCRError("OCR failed")

        # run_full_analysis 会捕获异常并返回 None
        result = self.service.run_full_analysis(pdf_path)
        self.assertIsNone(result)

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
        self.assertIn("### ⚠️ 合规风险提示", report)


if __name__ == "__main__":
    unittest.main()
