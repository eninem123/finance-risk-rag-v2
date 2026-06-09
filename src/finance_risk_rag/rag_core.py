"""
Finance-Risk-RAG 核心模块
========================

银行级多语言财务文本风控AI系统的核心RAG引擎。
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Protocol

import chromadb
from chromadb.utils import embedding_functions as ef

from .config import get_config
from .exceptions import DatabaseError, EmbeddingError, LLMError
from .models import ChunkConfig, DocumentChunk, EmbeddingBackend, QueryResult
from .utils import clean_text, ensure_dirs, split_text_by_sentence

# 配置日志
logger = logging.getLogger(__name__)


# ==================== 协议定义 ====================


class EmbeddingFunction(Protocol):
    """嵌入函数协议"""

    def __call__(self, texts: List[str]) -> List[List[float]]: ...


class LLMClient(Protocol):
    """LLM客户端协议"""

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str: ...


# ==================== 嵌入模型工厂 ====================


class EmbeddingModelFactory:
    """嵌入模型工厂类"""

    @staticmethod
    def create(
        backend: EmbeddingBackend = EmbeddingBackend.ONNX,
    ) -> Callable[[List[str]], List[List[float]]]:
        """
        创建嵌入函数
        """
        try:
            if backend == EmbeddingBackend.ONNX:
                return EmbeddingModelFactory._create_onnx_embedding()
            else:
                return EmbeddingModelFactory._create_sentence_transformer_embedding()
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            raise EmbeddingError(f"无法初始化嵌入模型: {e}") from e

    @staticmethod
    def _create_onnx_embedding() -> Callable[[List[str]], List[List[float]]]:
        """创建ONNX嵌入函数"""
        try:
            emb_fn = ef.ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
            logger.info("使用 ONNXMiniLM_L6_V2 作为嵌入函数")
            return emb_fn
        except Exception as e:
            logger.warning(f"ONNXMiniLM_L6_V2 不可用: {e}")
            raise

    @staticmethod
    def _create_sentence_transformer_embedding() -> Callable[[List[str]], List[List[float]]]:
        """创建SentenceTransformer嵌入函数"""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("使用 SentenceTransformer 作为嵌入函数")

        def embed(texts: List[str]) -> List[List[float]]:
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

        return embed


# ==================== LLM 客户端 ====================


class LLMClientWrapper:
    """LLM客户端封装类"""

    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_TOKENS = 512

    def __init__(self, config=None) -> None:
        self.config = config or get_config()
        self._client: Optional[Any] = None

        if not self.config.llm_api_key:
            logger.warning("未检测到 LLM API key。")
            return

        self._initialize_client()

    def _initialize_client(self) -> None:
        """初始化OpenAI兼容客户端"""
        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.config.llm_api_key, base_url=self.config.llm_base_url
            )
            logger.info(f"LLM客户端初始化成功，模型: {self.config.llm_model_name}")
        except Exception as e:
            logger.error(f"LLM客户端初始化失败: {e}")
            raise LLMError(f"无法初始化LLM客户端: {e}") from e

    @property
    def is_available(self) -> bool:
        """检查客户端是否可用"""
        return self._client is not None

    def ask(
        self,
        query: str,
        context: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """
        向LLM提问
        """
        if not self.is_available:
            raise LLMError("LLM客户端未初始化，请设置API密钥")

        system_prompt = "你是一名金融风险分析顾问，回答时引用上下文并给出简明结论。"
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"参考以下上下文来回答问题：\n\n{context}\n\n问题：{query}",
            },
        ]

        try:
            response = self._client.chat.completions.create(
                model=self.config.llm_model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise LLMError(f"LLM调用失败: {e}") from e


# ==================== 文本分块器 ====================


class TextChunker:
    """文本分块器"""

    def __init__(self, config: Optional[ChunkConfig] = None) -> None:
        self._config = config or ChunkConfig()

    def chunk(self, text: str) -> List[str]:
        """
        将文本分块
        """
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


# ==================== RAG 数据库 ====================


class RAGDatabase:
    """RAG向量数据库封装"""

    COLLECTION_NAME = "finance_docs"

    def __init__(self, config=None, embedding_fn: Optional[Callable] = None) -> None:
        self.config = config or get_config()
        ensure_dirs(self.config.chroma_db_dir)

        self._embedding_fn = embedding_fn or EmbeddingModelFactory.create()
        self._client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.Collection] = None

        self._initialize()

    def _initialize(self) -> None:
        """初始化数据库连接"""
        try:
            self._client = chromadb.PersistentClient(path=str(self.config.chroma_db_dir))
            self._collection = self._get_or_create_collection()
            logger.info(f"RAG数据库初始化成功: {self.config.chroma_db_dir}")
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise DatabaseError(f"无法初始化数据库: {e}") from e

    def _get_or_create_collection(self) -> chromadb.Collection:
        """获取或创建集合"""
        try:
            return self._client.get_collection(
                name=self.COLLECTION_NAME, embedding_function=self._embedding_fn
            )
        except Exception:
            return self._client.create_collection(
                name=self.COLLECTION_NAME, embedding_function=self._embedding_fn
            )

    def add_documents(self, chunks: List[DocumentChunk], batch_size: int = 100) -> int:
        """
        添加文档到数据库
        """
        if not chunks:
            return 0

        total_added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            documents = [chunk.content for chunk in batch]
            ids = [f"{chunk.source}__{chunk.chunk_index}" for chunk in batch]
            metadatas = [
                {"source": chunk.source, "chunk_index": chunk.chunk_index, **chunk.metadata}
                for chunk in batch
            ]

            try:
                self._collection.add(documents=documents, metadatas=metadatas, ids=ids)
                total_added += len(batch)
            except Exception as e:
                logger.error(f"添加文档失败: {e}")
                raise DatabaseError(f"添加文档失败: {e}") from e

        return total_added

    def query(self, query_text: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        查询相似文档
        """
        try:
            results = self._collection.query(query_texts=[query_text], n_results=top_k)

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            return [
                {"content": doc, "metadata": meta, "distance": dist}
                for doc, meta, dist in zip(documents, metadatas, distances)
            ]
        except Exception as e:
            logger.error(f"查询失败: {e}")
            raise DatabaseError(f"查询失败: {e}") from e


# ==================== RAG 引擎 ====================


class RAGEngine:
    """RAG引擎主类"""

    def __init__(self, config=None, chunk_config: Optional[ChunkConfig] = None) -> None:
        self.config = config or get_config()
        self._chunker = TextChunker(chunk_config)
        self._llm_client = LLMClientWrapper(self.config)
        self._database = RAGDatabase(self.config)

    def build_index(self) -> Dict[str, int]:
        """
        构建向量索引
        """
        stats = {"files_processed": 0, "chunks_added": 0, "errors": 0}

        if not self.config.docs_dir.exists():
            logger.warning(f"文档目录不存在: {self.config.docs_dir}")
            return stats

        txt_files = list(self.config.docs_dir.glob("*.txt"))

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
        """
        执行RAG查询
        """
        results = self._database.query(question, top_k=top_k)

        if not results:
            return QueryResult(answer="未找到相关文档。", sources=[], confidence=0.0)

        context = "\n\n".join(r["content"] for r in results)
        sources = [r["metadata"] for r in results]

        try:
            answer = self._llm_client.ask(question, context)
            return QueryResult(answer=answer, sources=sources, confidence=1.0)
        except LLMError as e:
            logger.error(f"LLM回答失败: {e}")
            return QueryResult(
                answer=f"无法生成回答: {e}",
                sources=sources,
                confidence=0.0,
                metadata={"error": str(e)},
            )
