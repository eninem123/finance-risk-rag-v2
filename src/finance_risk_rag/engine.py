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

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

import chromadb
from chromadb.utils import embedding_functions as ef

from .config import CHROMA_DB_DIR
from .llm import LLMClientWrapper
from .models import ChunkConfig, DocumentChunk, EmbeddingBackend, QueryResult
from .utils import clean_text, ensure_dirs, split_text_by_sentence

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 协议定义 ====================


class EmbeddingFunction(Protocol):
    """嵌入函数协议"""

    def __call__(self, texts: List[str]) -> List[List[float]]: ...


# ==================== 异常定义 ====================


class RAGError(Exception):
    """RAG系统基础异常"""


class EmbeddingError(RAGError):
    """嵌入模型相关异常"""


class LLMError(RAGError):
    """LLM调用相关异常"""


class DatabaseError(RAGError):
    """数据库相关异常"""


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
            emb_fn = ef.ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
            logger.info("使用 ONNXMiniLM_L6_V2 作为嵌入函数")
            return emb_fn
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
        db_path: str = CHROMA_DB_DIR,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ) -> None:
        """
        初始化RAG数据库

        Args:
            db_path: 数据库路径
            embedding_fn: 嵌入函数
        """
        ensure_dirs(db_path)

        self._db_path = db_path
        self._embedding_fn = embedding_fn or EmbeddingModelFactory.create()
        self._client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.Collection] = None

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

    def _get_or_create_collection(self) -> chromadb.Collection:
        """获取或创建集合"""
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

    def clear(self) -> None:
        """清空数据库"""
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
        docs_dir: str = "docs",
        db_path: str = CHROMA_DB_DIR,
        chunk_config: Optional[ChunkConfig] = None,
    ) -> None:
        """
        初始化RAG引擎

        Args:
            docs_dir: 文档目录
            db_path: 数据库路径
            chunk_config: 分块配置
        """
        self._docs_dir = Path(docs_dir)
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
