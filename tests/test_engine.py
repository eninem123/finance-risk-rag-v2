"""
RAG 引擎单元测试
"""

import unittest
from unittest.mock import MagicMock, patch

from src.finance_risk_rag.engine import RAGEngine


class TestRAGEngine(unittest.TestCase):
    @patch("chromadb.PersistentClient")
    def setUp(self, mock_chroma):
        self.mock_config = MagicMock()
        self.mock_config.chroma_db_dir = "test_db"
        self.mock_config.chunk_size = 800
        self.mock_llm = MagicMock()
        self.engine = RAGEngine(config=self.mock_config, llm_client=self.mock_llm)

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

    def test_add_documents_incremental(self):
        # Mock file hash and collection behavior
        mock_file = MagicMock()
        mock_file.name = "new_doc.txt"
        mock_file.read_text.return_value = "Content"

        with patch("src.finance_risk_rag.utils.get_file_hash", return_value="hash123"):
            # Case 1: Document doesn't exist
            self.engine._collection.get.return_value = {"metadatas": []}
            self.engine.add_documents([mock_file])
            self.assertTrue(self.engine._collection.add.called)

            # Case 2: Document exists with same hash
            self.engine._collection.add.reset_mock()
            self.engine._collection.get.return_value = {"metadatas": [{"hash": "hash123"}]}
            self.engine.add_documents([mock_file])
            self.assertFalse(self.engine._collection.add.called)


if __name__ == "__main__":
    unittest.main()
