"""
Finance-Risk-RAG RAG核心引擎
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions as ef
from openai import OpenAI

from finance_risk_rag.config import get_config
from finance_risk_rag.models import ChunkConfig, QueryResult, DocumentChunk
from finance_risk_rag.exceptions import RAGError, LLMError, DatabaseError
from finance_risk_rag.utils import clean_text, ensure_dirs, split_text_by_sentence


class LLMClientWrapper:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model_name: str = "moonshot-v1-8k") -> None:
        config = get_config()
        self._api_key = api_key or config.llm_api_key
        self._base_url = base_url or config.llm_base_url
        self._model_name = model_name
        self._client = None
        if self._api_key:
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    def ask(self, query: str, context: str) -> str:
        if not self._client:
            raise LLMError("LLM客户端未初始化")
        messages = [
            {"role": "system", "content": "你是一名金融风险分析顾问。"},
            {"role": "user", "content": f"参考上下文回答问题：\n\n{context}\n\n问题：{query}"}
        ]
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"LLM调用失败: {e}")


class RAGDatabase:
    def __init__(self, db_path: Path, embedding_fn: Optional[Callable] = None) -> None:
        self._db_path = db_path
        self._embedding_fn = embedding_fn or ef.ONNXMiniLM_L6_V2()
        self._client = chromadb.PersistentClient(path=str(db_path))
        self._collection = self._client.get_or_create_collection(
            name="finance_docs",
            embedding_function=self._embedding_fn
        )

    def add_documents(self, chunks: List[DocumentChunk]) -> int:
        if not chunks: return 0
        documents = [c.content for c in chunks]
        metadatas = [{"source": c.source, "index": c.chunk_index} for c in chunks]
        ids = [f"{c.source}_{c.chunk_index}" for c in chunks]
        self._collection.add(documents=documents, metadatas=metadatas, ids=ids)
        return len(chunks)

    def query(self, query_text: str, top_k: int = 4) -> List[Dict[str, Any]]:
        results = self._collection.query(query_texts=[query_text], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return [{"content": d, "metadata": m} for d, m in zip(docs, metas)]


class RAGEngine:
    def __init__(self, config: Optional[Any] = None) -> None:
        self._config = config or get_config()
        self._db = RAGDatabase(self._config.chroma_db_dir)
        self._llm = LLMClientWrapper()

    def build_index(self) -> Dict[str, int]:
        stats = {"files": 0, "chunks": 0}
        for txt_file in self._config.docs_dir.glob("*.txt"):
            if txt_file.name == "all_extracted.txt": continue
            content = txt_file.read_text(encoding="utf-8")
            text_chunks = split_text_by_sentence(content, max_len=self._config.chunk_size)
            chunks = [DocumentChunk(content=c, source=txt_file.name, chunk_index=i) for i, c in enumerate(text_chunks)]
            self._db.add_documents(chunks)
            stats["files"] += 1
            stats["chunks"] += len(chunks)
        return stats

    def query(self, question: str, top_k: int = 4) -> QueryResult:
        results = self._db.query(question, top_k=top_k)
        context = "\n\n".join(r["content"] for r in results)
        try:
            answer = self._llm.ask(question, context)
            return QueryResult(answer=answer, sources=[r["metadata"] for r in results])
        except Exception as e:
            return QueryResult(answer=f"错误: {e}", sources=[])
