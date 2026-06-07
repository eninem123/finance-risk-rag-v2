import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

import chromadb
from chromadb.utils import embedding_functions as ef

from finance_risk_rag.config import get_config
from finance_risk_rag.exceptions import DatabaseError, EmbeddingError
from finance_risk_rag.llm import LLMClientWrapper
from finance_risk_rag.models import ChunkConfig, DocumentChunk, QueryResult
from finance_risk_rag.utils import clean_text, ensure_dirs, split_text_by_sentence

logger = logging.getLogger(__name__)


class EmbeddingModelFactory:
    """嵌入模型工厂类"""

    @staticmethod
    def create(backend: str = "onnx") -> Callable[[List[str]], List[List[float]]]:
        try:
            if "onnx" in backend.lower():
                return EmbeddingModelFactory._create_onnx_embedding()
            else:
                return EmbeddingModelFactory._create_sentence_transformer_embedding()
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            raise EmbeddingError(f"无法初始化嵌入模型: {e}") from e

    @staticmethod
    def _create_onnx_embedding() -> Any:
        return ef.ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])

    @staticmethod
    def _create_sentence_transformer_embedding() -> Callable[[List[str]], List[List[float]]]:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        def embed(texts: List[str]) -> List[List[float]]:
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

        return embed


class TextChunker:
    """文本分块器"""

    def __init__(self, config: Optional[ChunkConfig] = None) -> None:
        self._config = config or ChunkConfig()

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []
        cleaned_text = clean_text(text)
        sentences = split_text_by_sentence(cleaned_text, max_len=self._config.chunk_size)

        chunks: List[str] = []
        for i, sentence in enumerate(sentences):
            if i == 0:
                current = sentence
            else:
                prev_sent = sentences[i - 1]
                overlap_part = (
                    prev_sent[-self._config.overlap :]
                    if len(prev_sent) >= self._config.overlap
                    else prev_sent
                )
                current = overlap_part + sentence

            if len(current) > self._config.chunk_size:
                current = current[: self._config.chunk_size]
            chunks.append(current)
        return chunks


class RAGDatabase:
    """RAG向量数据库封装"""

    COLLECTION_NAME = "finance_docs"

    def __init__(
        self, db_path: Path, embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None
    ) -> None:
        ensure_dirs(db_path)
        self._db_path = str(db_path)
        self._embedding_fn = embedding_fn or EmbeddingModelFactory.create()
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            self._client = chromadb.PersistentClient(path=self._db_path)
            self._collection = self._get_or_create_collection()
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise DatabaseError(f"无法初始化数据库: {e}") from e

    def _get_or_create_collection(self) -> Any:
        if self._client is None:
            raise DatabaseError("数据库客户端未初始化")
        try:
            return self._client.get_collection(
                name=self.COLLECTION_NAME, embedding_function=self._embedding_fn
            )
        except Exception:
            return self._client.create_collection(
                name=self.COLLECTION_NAME, embedding_function=self._embedding_fn
            )

    def add_documents(self, chunks: List[DocumentChunk], batch_size: int = 100) -> int:
        if not chunks or self._collection is None:
            return 0
        total_added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            documents = [chunk.content for chunk in batch]
            ids = [f"{chunk.source}__{chunk.chunk_index}" for chunk in batch]
            # Use cast to satisfy mypy's expectation for Mapping[str, str | int | float | bool ...]
            metadatas = cast(
                List[Dict[str, Any]],
                [
                    {"source": chunk.source, "chunk_index": chunk.chunk_index, **chunk.metadata}
                    for chunk in batch
                ],
            )
            try:
                self._collection.add(documents=documents, metadatas=metadatas, ids=ids)
                total_added += len(batch)
            except Exception as e:
                logger.error(f"添加文档失败: {e}")
                raise DatabaseError(f"添加文档失败: {e}") from e
        return total_added

    def query(self, query_text: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if self._collection is None:
            return []
        try:
            results = self._collection.query(query_texts=[query_text], n_results=top_k)
            documents = cast(List[List[str]], results.get("documents", [[]]))[0]
            metadatas = cast(List[List[Dict[str, Any]]], results.get("metadatas", [[]]))[0]
            distances = cast(List[List[float]], results.get("distances", [[]]))[0]
            return [
                {"content": doc, "metadata": meta, "distance": dist}
                for doc, meta, dist in zip(documents, metadatas, distances)
            ]
        except Exception as e:
            logger.error(f"查询失败: {e}")
            raise DatabaseError(f"查询失败: {e}") from e

    def clear(self) -> None:
        if self._client is None:
            return
        try:
            self._client.delete_collection(name=self.COLLECTION_NAME)
            self._collection = self._get_or_create_collection()
        except Exception as e:
            logger.error(f"清空数据库失败: {e}")
            raise DatabaseError(f"清空数据库失败: {e}") from e


class RAGEngine:
    """RAG引擎主类"""

    def __init__(
        self,
        docs_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
        chunk_config: Optional[ChunkConfig] = None,
    ) -> None:
        config = get_config()
        self._docs_dir = docs_dir or config.docs_dir
        self._chunker = TextChunker(chunk_config)
        self._llm_client = LLMClientWrapper()
        self._database = RAGDatabase(db_path or config.chroma_db_dir)

    def build_index(self) -> Dict[str, int]:
        stats = {"files_processed": 0, "chunks_added": 0, "errors": 0}
        if not self._docs_dir.exists():
            return stats

        txt_files = list(self._docs_dir.glob("*.txt"))
        for txt_file in txt_files:
            try:
                content = txt_file.read_text(encoding="utf-8")
                chunks = self._chunker.chunk(content)
                doc_chunks = [
                    DocumentChunk(content=chunk, source=txt_file.name, chunk_index=i)
                    for i, chunk in enumerate(chunks)
                ]
                added = self._database.add_documents(doc_chunks)
                stats["files_processed"] += 1
                stats["chunks_added"] += added
            except Exception as e:
                logger.error(f"处理文件失败 {txt_file}: {e}")
                stats["errors"] += 1
        return stats

    def query(self, question: str, top_k: int = 4) -> QueryResult:
        results = self._database.query(question, top_k=top_k)
        if not results:
            return QueryResult(answer="未找到相关文档。", sources=[], confidence=0.0)

        context = "\n\n".join(r["content"] for r in results)
        sources = [r["metadata"] for r in results]
        try:
            answer = self._llm_client.ask(question, context)
            return QueryResult(answer=answer, sources=sources, confidence=1.0)
        except Exception as e:
            logger.error(f"LLM回答失败: {e}")
            return QueryResult(
                answer=f"无法生成回答: {e}",
                sources=sources,
                confidence=0.0,
                metadata={"error": str(e)},
            )
