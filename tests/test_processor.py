"""
文档处理器单元测试
"""

import unittest
from unittest.mock import MagicMock, patch

from src.finance_risk_rag.processor import DocumentProcessor


class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock()
        self.mock_config.tesseract_cmd = "tesseract"
        self.mock_config.ocr_dpi = 300
        self.mock_config.ocr_languages = "eng"
        self.mock_config.ocr_version = "v1"
        self.mock_llm = MagicMock()
        self.processor = DocumentProcessor(config=self.mock_config, llm_client=self.mock_llm)

    def test_classify_document(self):
        # Mock LLM response
        self.mock_llm.chat.return_value = '{"type": "审计报告", "confidence": 0.95, "reason": "test"}'

        result = self.processor.classify_document("Sample financial text...")

        self.assertEqual(result.type, "审计报告")
        self.assertEqual(result.confidence, 0.95)

    @patch("src.finance_risk_rag.processor.ProcessPoolExecutor")
    def test_processor_parallel_execution(self, mock_executor):
        # This test ensures that parallel execution logic is called
        mock_dir = MagicMock()
        mock_pdf = MagicMock()
        mock_pdf.name = "test.pdf"
        mock_pdf.suffix = ".pdf"
        mock_pdf.with_suffix.return_value = MagicMock()
        mock_dir.glob.return_value = [mock_pdf]

        with patch.object(self.processor, "process_single_pdf") as mock_proc:
            mock_proc.return_value = {
                "name": "test.pdf",
                "text": "text",
                "classification": {"type": "A"},
                "hash": "h1",
                "ocr_pages": 0,
                "ocr_version": "v1",
            }
            # Set max_workers=1 to avoid real ProcessPool complexity in mock
            self.processor.process_directory(mock_dir, max_workers=1)
            self.assertTrue(mock_proc.called)


if __name__ == "__main__":
    unittest.main()
