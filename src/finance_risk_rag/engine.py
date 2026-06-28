"""
Finance-Risk-RAG RAG 引擎模块
============================

实现基于 ChromaDB 的检索增强生成 (RAG) 系统。
"""

import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions as ef

from .config import Config, get_config
from .exceptions import DatabaseError, RAGError
from .llm import LLMClientWrapper
from .models import QueryResult
from .utils import clean_text, ensure_dirs, split_text_by_sentence

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    RAG 引擎类，负责文档索引和语义检索问答。
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        llm_client: Optional[LLMClientWrapper] = None
    ):
        """
        初始化 RAG 引擎。

        Args:
            config: 配置对象。
            llm_client: LLM 客户端包装类。
        """
        self.config = config or get_config()
        self.llm_client = llm_client or LLMClientWrapper(
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
            model_name=self.config.llm_model_name,
        )
        self._db_path = self.config.chroma_db_dir
        ensure_dirs(self._db_path)

        self._initialize_db()

    def _initialize_db(self) -> None:
        """初始化向量数据库连接。"""
        try:
            self._client = chromadb.PersistentClient(path=str(self._db_path))
            # 默认使用 ONNX MiniLM 模型
            self._emb_fn = ef.ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
            self._collection = self._client.get_or_create_collection(
                name="finance_docs", embedding_function=self._emb_fn
            )
        except Exception as e:
            raise DatabaseError(f"Failed to initialize ChromaDB: {e}")

    def add_documents(self, txt_files: List[Path], force: bool = False) -> None:
        """
        向向量数据库添加文档，支持基于文件哈希的增量更新。

        Args:
            txt_files: 要添加的文本文件列表。
            force: 是否强制重新索引所有文件。
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
                    if existing and existing["metadatas"]:
                        if existing["metadatas"][0].get("hash") == current_hash:
                            logger.info(f"Skipping indexing for {txt_file.name} (unchanged)")
                            continue
                        else:
                            logger.info(f"Updating index for {txt_file.name} (content changed)")
                            self._collection.delete(where={"source": txt_file.name})

                content = txt_file.read_text(encoding="utf-8")
                cleaned = clean_text(content)
                sentences = split_text_by_sentence(cleaned, max_len=self.config.chunk_size)

                documents = []
                metadatas = []
                ids = []

                for i, sent in enumerate(sentences):
                    documents.append(sent)
                    metadatas.append(
                        {"source": txt_file.name, "chunk_index": i, "hash": current_hash}
                    )
                    ids.append(f"{txt_file.name}_{i}")

                if documents:
                    self._collection.add(documents=documents, metadatas=metadatas, ids=ids)
                    logger.info(f"Indexed {len(documents)} chunks from {txt_file.name}")
            except Exception as e:
                logger.error(f"Failed to index {txt_file}: {e}")

    def query(self, question: str, top_k: int = 4) -> QueryResult:
        """
        执行 RAG 查询。

        Args:
            question: 用户问题。
            top_k: 返回的相关文档块数量。

        Returns:
            QueryResult: 包含答案和来源的查询结果。

        Raises:
            RAGError: 查询或 LLM 调用失败。
        """
        try:
            results = self._collection.query(query_texts=[question], n_results=top_k)

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]

            context = "\n\n".join(docs)
            answer = self.llm_client.ask(question, context)

            return QueryResult(answer=answer, sources=metas, confidence=1.0)
        except Exception as e:
            raise RAGError(f"Query failed: {e}")

    def build_index(self) -> None:
        """从默认文档目录自动构建/更新索引。"""
        docs_dir = self.config.docs_dir
        txt_files = list(docs_dir.glob("*.txt"))
        txt_files = [f for f in txt_files if f.name not in ["all_extracted.txt"]]
        self.add_documents(txt_files)
