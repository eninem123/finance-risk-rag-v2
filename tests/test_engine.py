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

    def test_query(self):
        # Setup mock collection results
        self.engine._collection.query.return_value = {
            "documents": [["context fragment"]],
            "metadatas": [[{"source": "doc.txt"}]],
        }
        self.mock_llm.ask.return_value = "Expert answer"

        result = self.engine.query("test question")

        self.assertEqual(result.answer, "Expert answer")
        self.assertEqual(result.sources[0]["source"], "doc.txt")

        # Test case 2: Unchanged content
        from src.finance_risk_rag.utils import get_file_hash

        file_hash = get_file_hash(txt_file)
        mock_collection.get.return_value = {"metadatas": [{"hash": file_hash}]}
        mock_collection.add.reset_mock()

        with patch("src.finance_risk_rag.utils.get_file_hash", return_value="hash123"):
            # Case 1: Document doesn't exist
            self.engine._collection.get.return_value = {"metadatas": []}
            self.engine.add_documents([mock_file])
            self.assertTrue(self.engine._collection.add.called)

        # Test case 3: Changed content
        txt_file.write_text("Changed content")
        mock_collection.get.return_value = {"metadatas": [{"hash": "old_hash"}]}


if __name__ == "__main__":
    unittest.main()
