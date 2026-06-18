from unittest.mock import MagicMock, patch

import pytest

from src.finance_risk_rag.engine import RAGEngine


@pytest.fixture
def mock_engine_config(tmp_path):
    conf = MagicMock()
    conf.chroma_db_dir = tmp_path / "db"
    conf.docs_dir = tmp_path / "docs"
    conf.docs_dir.mkdir()
    conf.chunk_size = 100
    conf.llm_api_key = "test"
    return conf


def test_engine_incremental_indexing(mock_engine_config):
    # Mock chromadb
    with patch("chromadb.PersistentClient"), patch(
        "chromadb.utils.embedding_functions.ONNXMiniLM_L6_V2"
    ), patch("src.finance_risk_rag.engine.LLMClientWrapper"):

        engine = RAGEngine(config=mock_engine_config)
        mock_collection = MagicMock()
        engine._collection = mock_collection

        txt_file = mock_engine_config.docs_dir / "test.txt"
        txt_file.write_text("Finance content")

        # Test case 1: First time indexing
        mock_collection.get.return_value = {"metadatas": []}
        engine.add_documents([txt_file])
        assert mock_collection.add.call_count == 1

        # Test case 2: Unchanged content
        from src.finance_risk_rag.utils import get_file_hash

        file_hash = get_file_hash(txt_file)
        mock_collection.get.return_value = {"metadatas": [{"hash": file_hash}]}
        mock_collection.add.reset_mock()

        engine.add_documents([txt_file])
        assert mock_collection.add.call_count == 0

        # Test case 3: Changed content
        txt_file.write_text("Changed content")
        new_hash = get_file_hash(txt_file)
        mock_collection.get.return_value = {"metadatas": [{"hash": "old_hash"}]}

        engine.add_documents([txt_file])
        assert mock_collection.delete.called
        assert mock_collection.add.call_count == 1
