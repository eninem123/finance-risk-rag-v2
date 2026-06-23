"""
Finance-Risk-RAG 终端到终端集成测试
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.finance_risk_rag.service import RiskAnalysisService
from src.finance_risk_rag.models import Entity, ExtractionResult

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock()
        self.mock_config.docs_dir = Path("test_docs")
        self.mock_config.cache_dir = Path("test_cache")
        self.mock_config.ocr_version = "v1"
        self.mock_config.llm_api_key = "test-key"
        self.mock_config.llm_base_url = "https://api.test.com"
        self.mock_config.llm_model_name = "test-model"

        # Mock dependencies to avoid external calls
        self.mock_processor = MagicMock()
        self.mock_pipeline = MagicMock()
        self.mock_engine = MagicMock()

        # Patch RAGEngine initialization
        with patch("src.finance_risk_rag.service.RAGEngine", return_value=self.mock_engine):
            self.service = RiskAnalysisService(
                config=self.mock_config,
                processor=self.mock_processor,
                pipeline=self.mock_pipeline
            )

    @patch("pathlib.Path.is_file", return_value=True)
    def test_full_workflow(self, mock_is_file):
        """测试从 PDF 处理到报告生成的完整流程"""
        pdf_path = Path("sample.pdf")

        # 1. Mock Processor
        self.mock_processor.process_single_pdf.return_value = {
            "name": "sample.pdf",
            "text": "This is a sample document with high debt.",
            "classification": {"type": "审计报告", "confidence": 0.95},
            "hash": "hash123"
        }

        # 2. Mock Pipeline
        mock_entities = [
            Entity(type="CREDIT_RISK", text="high debt", risk_score=30, confidence=0.9, source="bert")
        ]
        self.mock_pipeline.process.return_value = ExtractionResult(
            entities=mock_entities, total_risk_score=30, risk_level="低风险"
        )

        # 3. 执行全流程分析
        output = self.service.run_full_analysis(pdf_path)

        self.assertEqual(output["status"], "success")
        self.assertEqual(output["count"], 1)
        analysis_data = output["results"][0]

        # 4. 生成报告
        # Mock LLM for executive summary
        self.mock_processor.llm_client.is_available = True
        self.mock_processor.llm_client.chat.return_value = "This is a professional executive summary."

        report = self.service.generate_report(analysis_data)

        # 5. 验证报告内容
        self.assertIn("# 财务风险分析报告: sample.pdf", report)
        self.assertIn("## 0. 执行摘要", report)
        self.assertIn("This is a professional executive summary.", report)
        self.assertIn("审计报告", report)
        self.assertIn("CREDIT_RISK", report)
        self.assertIn("high debt", report)

if __name__ == "__main__":
    unittest.main()
