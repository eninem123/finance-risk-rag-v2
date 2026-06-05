"""
RAG core engine for indexing and querying.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import chromadb
from chromadb.utils import embedding_functions as ef

from .config import get_config
from .llm import LLMClientWrapper
from .models import QueryResult
from .utils import clean_text, split_text_by_sentence


class RAGEngine:
    """Core RAG engine for vector search and generation."""

    def __init__(self, config=None) -> None:
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        self.llm = LLMClientWrapper()

        # Initialize Chroma
        self.config.ensure_directories()
        self.db_client = chromadb.PersistentClient(path=str(self.config.chroma_db_dir))

        # Use ONNX MiniLM as default embedding function
        self.emb_fn = ef.ONNXMiniLM_L6_V2()

        self.collection = self.db_client.get_or_create_collection(
            name="finance_risk", embedding_function=self.emb_fn
        )

    def build_index(self, docs_dir: Optional[Path] = None) -> Dict[str, int]:
        """Chunk documents and add them to the vector store."""
        docs_dir = docs_dir or self.config.docs_dir
        txt_files = list(docs_dir.glob("*.txt"))

        total_chunks = 0
        for txt_file in txt_files:
            if txt_file.name == "all_extracted.txt":
                continue

            content = txt_file.read_text(encoding="utf-8")
            chunks = split_text_by_sentence(clean_text(content), max_len=self.config.chunk_size)

            documents = []
            metadatas = []
            ids = []

            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({"source": txt_file.name, "index": i})
                ids.append(f"{txt_file.name}_{i}")
                total_chunks += 1

            if documents:
                self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

        return {"processed_files": len(txt_files), "total_chunks": total_chunks}

    def query(self, question: str, top_k: int = 5) -> QueryResult:
        """Query the RAG system."""
        results = self.collection.query(query_texts=[question], n_results=top_k)

        # Flatten results
        docs = results["documents"][0]
        metas = results["metadatas"][0]

        context = "\n\n".join(docs)
        answer = self.llm.ask(question, context)

        return QueryResult(answer=answer, sources=metas, confidence=1.0)
