"""
Finance-Risk-RAG 核心引擎
========================
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import chromadb
from chromadb.utils import embedding_functions as ef

from .config import get_config
from .utils import ensure_dirs, setup_logger, split_text_by_sentence

logger = setup_logger("rag_core", "logs/rag_core_optimized.log")


class EmbeddingBackend(Enum):
    ONNX = "onnx"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


@dataclass
class QueryResult:
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    content: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMClientWrapper:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        config = get_config()
        self._api_key = api_key or config.llm_api_key
        self._base_url = base_url or config.llm_base_url
        self._model_name = model_name or config.llm_model_name
        self._client: Optional[Any] = None

        if self._api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    def call(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if not self._client:
            return "LLM Client not initialized"
        try:
            response = self._client.chat.completions.create(
                model=self._model_name, messages=messages, **kwargs
            )
            content = response.choices[0].message.content
            return content if content else ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Error: {e}"


class RAGEngine:
    def __init__(
        self,
        docs_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ) -> None:
        config = get_config()
        self.docs_dir = docs_dir or config.docs_dir
        self.db_path = db_path or config.chroma_db_dir
        self.llm = LLMClientWrapper()

        ensure_dirs(self.db_path)
        self.client = chromadb.PersistentClient(path=str(self.db_path))

        # Default embedding function
        self.emb_fn = ef.ONNXMiniLM_L6_V2()

        try:
            self.collection = self.client.get_collection(name="finance_docs")
        except Exception:
            self.collection = self.client.create_collection(
                name="finance_docs", embedding_function=cast(Any, self.emb_fn)
            )

    def build_index(self) -> Dict[str, int]:
        stats = {"files": 0, "chunks": 0}
        for txt_file in self.docs_dir.glob("*.txt"):
            content = txt_file.read_text(encoding="utf-8")
            sentences = split_text_by_sentence(content)

            ids = []
            docs = []
            metas = []

            for i, sent in enumerate(sentences):
                ids.append(f"{txt_file.name}_{i}")
                docs.append(sent)
                metas.append({"source": txt_file.name, "index": i})

            if ids:
                self.collection.add(ids=ids, documents=docs, metadatas=metas)
                stats["files"] += 1
                stats["chunks"] += len(ids)

        return stats

    def query(self, question: str, top_k: int = 4) -> QueryResult:
        results = self.collection.query(query_texts=[question], n_results=top_k)

        docs = results.get("documents")
        metas = results.get("metadatas")

        if not docs or not docs[0]:
            return QueryResult("No relevant documents found", [])

        context = "\n\n".join(docs[0])
        messages = [
            {"role": "system", "content": "You are a financial risk analyst."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]

        answer = self.llm.call(messages)
        return QueryResult(answer, metas[0] if metas else [])
