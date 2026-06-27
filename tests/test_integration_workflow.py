"""
端到端集成工作流测试
"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from src.finance_risk_rag.config import Config
from src.finance_risk_rag.service import RiskAnalysisService


class TestIntegrationWorkflow(unittest.TestCase):
    def setUp(self):
        # Setup test environment variables
        os.environ["MOONSHOT_API_KEY"] = "test_key"
        self.config = Config(
            base_dir=Path("/tmp/finance_risk_test"),
        )
        self.config.ensure_directories()

    @patch("src.finance_risk_rag.processor.DocumentProcessor.extract_text_from_pdf")
    @patch("src.finance_risk_rag.llm.LLMClientWrapper.chat")
    def test_full_workflow(self, mock_chat, mock_extract):
        # 1. Mock OCR output
        mock_extract.return_value = ("Sample financial document text with high debt.", 0)

        # 2. Mock LLM classification and summary
        def mock_chat_responses(messages, **kwargs):
            content = messages[-1]["content"]  # Get the last message which contains user prompt
            if "判断以下财务文档属于哪一类" in content:
                return '{"type": "审计报告", "confidence": 0.95, "reason": "Test"}'
            if "风险执行摘要" in content:
                return "This is a test executive summary."
            return "Default response"

        mock_chat.side_effect = mock_chat_responses

        # 3. Initialize Service
        service = RiskAnalysisService(config=self.config)

        # 4. Create a dummy PDF file
        pdf_path = self.config.docs_dir / "test_doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 dummy content")

        # 5. Run analysis
        analysis = service.analyze_document(pdf_path)

        # 6. Verifications
        self.assertEqual(analysis["classification"]["type"], "审计报告")
        self.assertEqual(analysis["executive_summary"], "This is a test executive summary.")
        self.assertIn("risk_analysis", analysis)

        # 7. Generate report
        report_path = self.config.base_dir / "report.md"
        report = service.generate_report(analysis, report_path)

        self.assertTrue(report_path.exists())
        self.assertIn("# 财务风险分析报告: test_doc.pdf", report)
        self.assertIn("This is a test executive summary.", report)


if __name__ == "__main__":
    unittest.main()
