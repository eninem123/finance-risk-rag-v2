"""
Finance-Risk-RAG RAG 引擎模块
============================
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.utils import embedding_functions as ef

from .config import get_config
from .exceptions import DatabaseError, RAGError
from .llm import LLMClientWrapper
from .models import ChunkConfig, DocumentChunk, QueryResult
from .utils import clean_text, ensure_dirs, split_text_by_sentence

logger = logging.getLogger(__name__)

class RAGEngine:
    """RAG 引擎主类"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.llm_client = LLMClientWrapper()
        self._db_path = self.config.chroma_db_dir
        ensure_dirs(self._db_path)

        self._initialize_db()

    def _initialize_db(self):
        try:
            self._client = chromadb.PersistentClient(path=str(self._db_path))
            # Use ONNX as default embedding function
            self._emb_fn = ef.ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
            self._collection = self._client.get_or_create_collection(
                name="finance_docs",
                embedding_function=self._emb_fn
            )
        except Exception as e:
            raise DatabaseError(f"Failed to initialize ChromaDB: {e}")

    def add_documents(self, txt_files: List[Path]):
        for txt_file in txt_files:
            try:
                content = txt_file.read_text(encoding="utf-8")
                cleaned = clean_text(content)
                sentences = split_text_by_sentence(cleaned, max_len=self.config.chunk_size)

                documents = []
                metadatas = []
                ids = []

                for i, sent in enumerate(sentences):
                    documents.append(sent)
                    metadatas.append({"source": txt_file.name, "chunk_index": i})
                    ids.append(f"{txt_file.name}_{i}")

                if documents:
                    self._collection.add(documents=documents, metadatas=metadatas, ids=ids)
                    logger.info(f"Added {len(documents)} chunks from {txt_file.name}")
            except Exception as e:
                logger.error(f"Failed to index {txt_file}: {e}")

    def query(self, question: str, top_k: int = 4) -> QueryResult:
        try:
            results = self._collection.query(
                query_texts=[question],
                n_results=top_k
            )

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]

            context = "\n\n".join(docs)
            answer = self.llm_client.ask(question, context)

            return QueryResult(
                answer=answer,
                sources=metas,
                confidence=1.0
            )
        except Exception as e:
            raise RAGError(f"Query failed: {e}")

    def build_index(self):
        docs_dir = self.config.docs_dir
        txt_files = list(docs_dir.glob("*.txt"))
        # Exclude aggregate files
        txt_files = [f for f in txt_files if f.name not in ["all_extracted.txt"]]
        self.add_documents(txt_files)
