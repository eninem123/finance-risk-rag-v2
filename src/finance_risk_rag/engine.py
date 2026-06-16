"""
Finance-Risk-RAG RAG 引擎模块
============================
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, cast

import chromadb
from chromadb.utils import embedding_functions as ef

from .config import get_config
from .exceptions import DatabaseError, RAGError
from .llm import LLMClientWrapper
from .models import QueryResult
from .utils import clean_text, ensure_dirs, split_text_by_sentence

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG 引擎主类"""

    def __init__(self, config=None, llm_client=None):
        self.config = config or get_config()
        self.llm_client = llm_client or LLMClientWrapper(
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
            model_name=self.config.llm_model_name,
        )
        self._db_path = self.config.chroma_db_dir
        ensure_dirs(self._db_path)

        self._initialize_db()

    def _initialize_db(self):
        try:
            self._client = chromadb.PersistentClient(path=str(self._db_path))
            # Use ONNX as default embedding function
            self._emb_fn = ef.ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
            self._collection = self._client.get_or_create_collection(
                name="finance_docs", embedding_function=self._emb_fn
            )
        except Exception as e:
            raise DatabaseError(f"Failed to initialize ChromaDB: {e}")

    def add_documents(self, txt_files: List[Path], force: bool = False):
        """
        向向量数据库添加文档，支持增量更新。
        """
        for txt_file in txt_files:
            try:
                from .utils import get_file_hash

                current_hash = get_file_hash(txt_file)

                # 检查是否已索引且未改变
                if not force:
                    existing = self._collection.get(
                        where={"source": txt_file.name}, include=["metadatas"]
                    )
                    metadatas = cast(List[Dict[str, Any]], existing.get("metadatas", []))
                    if metadatas:
                        # 检查第一个 chunk 的 hash 是否一致
                        if metadatas[0].get("hash") == current_hash:
                            logger.info(f"Skipping indexing for {txt_file.name} (unchanged)")
                            continue
                        else:
                            # 如果已改变，先删除旧的索引
                            logger.info(f"Updating index for {txt_file.name} (content changed)")
                            self._collection.delete(where={"source": txt_file.name})

                content = txt_file.read_text(encoding="utf-8")
                cleaned = clean_text(content)
                sentences = split_text_by_sentence(cleaned, max_len=self.config.chunk_size)

                documents = []
                metadatas_to_add = []
                ids = []

                for i, sent in enumerate(sentences):
                    documents.append(sent)
                    metadatas_to_add.append(
                        {"source": txt_file.name, "chunk_index": i, "hash": current_hash}
                    )
                    ids.append(f"{txt_file.name}_{i}")

                if documents:
                    self._collection.add(documents=documents, metadatas=metadatas_to_add, ids=ids)
                    logger.info(f"Indexed {len(documents)} chunks from {txt_file.name}")
            except Exception as e:
                logger.error(f"Failed to index {txt_file}: {e}")

    def query(self, question: str, top_k: int = 4) -> QueryResult:
        try:
            results = self._collection.query(query_texts=[question], n_results=top_k)

            documents = results.get("documents", [[]])
            metadatas = results.get("metadatas", [[]])

            if not documents or not documents[0]:
                return QueryResult(answer="未找到相关信息。", sources=[], confidence=0.0)

            docs = documents[0]
            metas = cast(List[Dict[str, Any]], metadatas[0])

            context = "\n\n".join(docs)
            answer = self.llm_client.ask(question, context)

            return QueryResult(answer=answer, sources=metas, confidence=1.0)
        except Exception as e:
            raise RAGError(f"Query failed: {e}")

    def build_index(self):
        docs_dir = self.config.docs_dir
        txt_files = list(docs_dir.glob("*.txt"))
        # Exclude aggregate files
        txt_files = [f for f in txt_files if f.name not in ["all_extracted.txt"]]
        self.add_documents(txt_files)
