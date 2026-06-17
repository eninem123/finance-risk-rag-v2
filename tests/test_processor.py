from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.finance_risk_rag.config import Config
from src.finance_risk_rag.processor import DocumentProcessor


@pytest.fixture
def mock_config(tmp_path):
    conf = Config()
    conf.docs_dir = tmp_path / "docs"
    conf.cache_dir = tmp_path / "cache"
    conf.docs_dir.mkdir()
    conf.cache_dir.mkdir()
    return conf


def test_processor_parallel_execution(mock_config):
    # Create mock PDF files
    (mock_config.docs_dir / "test1.pdf").write_text("dummy")
    (mock_config.docs_dir / "test2.pdf").write_text("dummy")

    processor = DocumentProcessor(config=mock_config)

    # Mock extract_text_from_pdf and classify_document
    with patch.object(
        DocumentProcessor, "extract_text_from_pdf", return_value=("extracted text", 1)
    ), patch.object(
        DocumentProcessor,
        "classify_document",
        return_value=MagicMock(to_dict=lambda: {"type": "Report"}),
    ):

        processor.process_directory(max_workers=2)

        assert (mock_config.docs_dir / "test1.txt").exists()
        assert (mock_config.docs_dir / "test2.txt").exists()
        assert (mock_config.docs_dir / "all_extracted.txt").exists()
        assert (mock_config.docs_dir / "classification.json").exists()
