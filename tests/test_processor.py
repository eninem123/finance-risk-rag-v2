from unittest.mock import MagicMock, patch

import pytest

from src.finance_risk_rag.config import Config
from src.finance_risk_rag.processor import DocumentProcessor




def test_processor_parallel_execution(mock_config):
    # Create mock PDF files
    (mock_config.docs_dir / "test1.pdf").write_text("dummy")
    (mock_config.docs_dir / "test2.pdf").write_text("dummy")

    def test_classify_document(self):
        # Mock LLM response
        self.mock_llm.chat.return_value = (
            '{"type": "审计报告", "confidence": 0.95, "reason": "test"}'
        )

    # Mock extract_text_from_pdf and classify_document
    with patch.object(
        DocumentProcessor, "extract_text_from_pdf", return_value=("extracted text", 1)
    ), patch.object(
        DocumentProcessor,
        "classify_document",
        return_value=MagicMock(to_dict=lambda: {"type": "Report"}),
    ):

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
