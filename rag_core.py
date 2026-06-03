"""
Finance-Risk-RAG 核心模块
========================

银行级多语言财务文本风控AI系统的核心RAG引擎。

功能:
    - 向量化文档并存储到Chroma数据库
    - 基于语义相似度的文档检索
    - 集成LLM进行智能问答

作者: Finance-Risk-RAG Team
版本: 2.0.0
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, cast

import chromadb
from chromadb.utils import embedding_functions as ef

from config import get_config
from utils import clean_text, ensure_dirs, setup_logger, split_text_by_sentence

# 配置日志
logger = setup_logger("rag_core", "logs/rag_core.log")


# ==================== 数据类定义 ====================


class EmbeddingBackend(Enum):
    """嵌入模型后端枚举"""

    ONNX = "onnx"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


@dataclass
class ChunkConfig:
    """文本分块配置"""

    chunk_size: int = 800
    overlap: int = 100

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if self.overlap < 0:
            raise ValueError("overlap 不能为负数")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap 必须小于 chunk_size")


@dataclass
class QueryResult:
    """查询结果数据类"""

    answer: str
    sources: List[Dict[str, Any]]
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """文档分块数据类"""

    content: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== 协议定义 ====================


class EmbeddingFunction(Protocol):
    """嵌入函数协议"""

    def __call__(self, texts: List[str]) -> List[List[float]]: ...


class LLMClient(Protocol):
    """LLM客户端协议"""

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str: ...


# ==================== 异常定义 ====================


class RAGError(Exception):
    """RAG系统基础异常"""

    pass


class EmbeddingError(RAGError):
    """嵌入模型相关异常"""

    pass


class LLMError(RAGError):
    """LLM调用相关异常"""

    pass


class DatabaseError(RAGError):
    """数据库相关异常"""

    pass


# ==================== 嵌入模型工厂 ====================


class EmbeddingModelFactory:
    """嵌入模型工厂类"""

    @staticmethod
    def create(
        backend: EmbeddingBackend = EmbeddingBackend.ONNX,
    ) -> Callable[[List[str]], List[List[float]]]:
        """
        创建嵌入函数

        Args:
            backend: 嵌入模型后端类型

        Returns:
            嵌入函数

        Raises:
            EmbeddingError: 嵌入模型初始化失败
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
            # ONNXMiniLM_L6_V2 might not be directly in ef or have different signature
            # Using cast to avoid mypy issues with dynamically loaded libraries
            onnx_ef: Any = ef
            emb_fn = onnx_ef.ONNXMiniLM_L6_V2()
            logger.info("使用 ONNXMiniLM_L6_V2 作为嵌入函数")
            return cast(Callable[[List[str]], List[List[float]]], emb_fn)
        except Exception as e:
            logger.warning(f"ONNXMiniLM_L6_V2 不可用: {e}，尝试备用方案")
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

    DEFAULT_MODEL = "moonshot-v1-8k"
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_TOKENS = 512

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """
        初始化LLM客户端

        Args:
            api_key: API密钥
            base_url: API基础URL
            model_name: 模型名称

        Raises:
            LLMError: 客户端初始化失败
        """
        config = get_config()
        self._api_key = api_key or config.llm_api_key
        self._base_url = base_url or config.llm_base_url
        self._model_name = model_name or config.llm_model_name
        self._client: Optional[Any] = None

        if not self._api_key:
            logger.warning(
                "未检测到 LLM API key。请设置环境变量 OPENAI_API_KEY 或 MOONSHOT_API_KEY。"
            )
            return

        self._initialize_client()

    def _initialize_client(self) -> None:
        """初始化OpenAI兼容客户端"""
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
            logger.info(f"LLM客户端初始化成功，模型: {self._model_name}")
        except Exception as e:
            logger.error(f"LLM客户端初始化失败: {e}")
            raise LLMError(f"无法初始化LLM客户端: {e}") from e

    @property
    def is_available(self) -> bool:
        """检查客户端是否可用"""
        return self._client is not None

    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """
        通用的LLM调用方法

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            LLM回答内容

        Raises:
            LLMError: LLM调用失败
        """
        if not self._client:
            raise LLMError("LLM客户端未初始化，请设置API密钥")

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return content if content else ""
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise LLMError(f"LLM调用失败: {e}") from e

    def ask(
        self,
        query: str,
        context: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """
        向LLM提问（专用于RAG问答）

        Args:
            query: 用户问题
            context: 上下文内容
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            LLM回答
        """
        system_prompt = "你是一名金融风险分析顾问，回答时引用上下文并给出简明结论。"
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"参考以下上下文来回答问题：\n\n{context}\n\n问题：{query}",
            },
        ]
        return self.call(messages, temperature, max_tokens)


# ==================== 文本分块器 ====================


class TextChunker:
    """文本分块器"""

    def __init__(self, config: Optional[ChunkConfig] = None) -> None:
        """
        初始化分块器

        Args:
            config: 分块配置
        """
        self._config = config or ChunkConfig()

    def chunk(self, text: str) -> List[str]:
        """
        将文本分块

        Args:
            text: 输入文本

        Returns:
            分块后的文本列表
        """
        if not text:
            return []

        # 清洗文本
        cleaned_text = clean_text(text)

        # 按句子拆分
        sentences = split_text_by_sentence(cleaned_text, max_len=self._config.chunk_size)

        # 处理重叠
        chunks: List[str] = []
        for i, sentence in enumerate(sentences):
            if i == 0:
                current = sentence
            else:
                # 重叠前一个句子的最后部分
                prev_sent = sentences[i - 1]
                overlap_part = (
                    prev_sent[-self._config.overlap :]
                    if len(prev_sent) >= self._config.overlap
                    else prev_sent
                )
                current = overlap_part + sentence

            # 确保单chunk不超过最大长度
            if len(current) > self._config.chunk_size:
                current = current[: self._config.chunk_size]

            chunks.append(current)

        return chunks


# ==================== RAG 数据库 ====================


class RAGDatabase:
    """RAG向量数据库封装"""

    COLLECTION_NAME = "finance_docs"

    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ) -> None:
        """
        初始化RAG数据库

        Args:
            db_path: 数据库路径
            embedding_fn: 嵌入函数
        """
        config = get_config()
        self._db_path = db_path or str(config.chroma_db_dir)
        ensure_dirs(self._db_path)
        self._embedding_fn = embedding_fn or EmbeddingModelFactory.create()
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None

        self._initialize()

    def _initialize(self) -> None:
        """初始化数据库连接"""
        try:
            self._client = chromadb.PersistentClient(path=self._db_path)
            self._collection = self._get_or_create_collection()
            logger.info(f"RAG数据库初始化成功: {self._db_path}")
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise DatabaseError(f"无法初始化数据库: {e}") from e

    def _get_or_create_collection(self) -> Any:
        """获取或创建集合"""
        if self._client is None:
            return None
        try:
            return self._client.get_collection(name=self.COLLECTION_NAME)
        except Exception:
            return self._client.create_collection(
                name=self.COLLECTION_NAME, embedding_function=self._embedding_fn
            )

    def add_documents(self, chunks: List[DocumentChunk], batch_size: int = 100) -> int:
        """
        添加文档到数据库

        Args:
            chunks: 文档分块列表
            batch_size: 批量处理大小

        Returns:
            添加的文档数量
        """
        if not chunks or not self._collection:
            return 0

        total_added = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            documents = [chunk.content for chunk in batch]
            ids = [f"{chunk.source}__{chunk.chunk_index}" for chunk in batch]
            metadatas = [
                {"source": chunk.source, "chunk_index": chunk.chunk_index} for chunk in batch
            ]

            try:
                self._collection.add(documents=documents, metadatas=metadatas, ids=ids)
                total_added += len(batch)
                logger.info(f"已添加 {len(batch)} 个文档块到数据库")
            except Exception as e:
                logger.error(f"添加文档失败: {e}")
                raise DatabaseError(f"添加文档失败: {e}") from e

        return total_added

    def query(self, query_text: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        查询相似文档

        Args:
            query_text: 查询文本
            top_k: 返回结果数量

        Returns:
            相似文档列表
        """
        if not self._collection:
            return []
        try:
            results = self._collection.query(query_texts=[query_text], n_results=top_k)

            documents = results.get("documents")
            metadatas = results.get("metadatas")
            distances = results.get("distances")

            if documents and metadatas and distances:
                return [
                    {"content": doc, "metadata": meta, "distance": dist}
                    for doc, meta, dist in zip(documents[0], metadatas[0], distances[0])
                ]
            return []
        except Exception as e:
            logger.error(f"查询失败: {e}")
            raise DatabaseError(f"查询失败: {e}") from e

    def clear(self) -> None:
        """清空数据库"""
        if not self._client:
            return
        try:
            self._client.delete_collection(name=self.COLLECTION_NAME)
            self._collection = self._get_or_create_collection()
            logger.info("数据库已清空")
        except Exception as e:
            logger.error(f"清空数据库失败: {e}")
            raise DatabaseError(f"清空数据库失败: {e}") from e


# ==================== RAG 引擎 ====================


class RAGEngine:
    """RAG引擎主类"""

    def __init__(
        self,
        docs_dir: Optional[str] = None,
        db_path: Optional[str] = None,
        chunk_config: Optional[ChunkConfig] = None,
    ) -> None:
        """
        初始化RAG引擎

        Args:
            docs_dir: 文档目录
            db_path: 数据库路径
            chunk_config: 分块配置
        """
        config = get_config()
        self._docs_dir = Path(docs_dir) if docs_dir else config.docs_dir
        self._chunker = TextChunker(chunk_config)
        self._llm_client = LLMClientWrapper()
        self._database = RAGDatabase(db_path)

    def build_index(self) -> Dict[str, int]:
        """
        构建向量索引

        Returns:
            构建统计信息
        """
        stats = {"files_processed": 0, "chunks_added": 0, "errors": 0}

        if not self._docs_dir.exists():
            logger.warning(f"文档目录不存在: {self._docs_dir}")
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

        logger.info(f"索引构建完成: {stats}")
        return stats

    def query(self, question: str, top_k: int = 4) -> QueryResult:
        """
        执行RAG查询

        Args:
            question: 用户问题
            top_k: 检索文档数量

        Returns:
            查询结果
        """
        # 检索相关文档
        results = self._database.query(question, top_k=top_k)

        if not results:
            return QueryResult(answer="未找到相关文档。", sources=[], confidence=0.0)

        # 组装上下文
        context = "\n\n".join(r["content"] for r in results)
        sources = [r["metadata"] for r in results]

        # 调用LLM生成回答
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


# ==================== 命令行入口 ====================


def main() -> None:
    """命令行入口函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Finance-Risk-RAG 核心引擎")
    parser.add_argument("--build-db", action="store_true", help="构建向量数据库")
    parser.add_argument("--query", type=str, help="执行查询")
    parser.add_argument("--top-k", type=int, default=4, help="返回结果数量")

    args = parser.parse_args()

    engine = RAGEngine()

    if args.build_db:
        stats = engine.build_index()
        print(f"索引构建完成: {stats}")

    if args.query:
        result = engine.query(args.query, top_k=args.top_k)
        print(f"回答: {result.answer}")
        print(f"来源: {result.sources}")


if __name__ == "__main__":
    main()
