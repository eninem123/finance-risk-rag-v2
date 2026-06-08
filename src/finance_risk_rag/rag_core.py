"""
Finance-Risk-RAG RAG 核心引擎
"""

from typing import Dict

import chromadb
from chromadb.utils import embedding_functions as ef
from openai import OpenAI

from finance_risk_rag.config import get_config
from finance_risk_rag.models import QueryResult
from finance_risk_rag.utils import (
    clean_text,
    ensure_dirs,
    setup_logger,
    split_text_by_sentence,
)


class LLMClientWrapper:
    """LLM 客户端封装"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.client = None
        if self.config.llm_api_key:
            self.client = OpenAI(api_key=self.config.llm_api_key, base_url=self.config.llm_base_url)

    def ask(self, query: str, context: str) -> str:
        if not self.client:
            return "LLM 客户端未配置"

        system_prompt = "你是一名金融风险分析顾问，回答时引用上下文并给出简明结论。"
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"参考以下上下文来回答问题：\n\n{context}\n\n问题：{query}",
            },
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model_name, messages=messages, temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM 调用失败: {e}"


class RAGEngine:
    """RAG 引擎主类"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = setup_logger("rag_engine", str(self.config.log_dir / "rag.log"))

        self.llm = LLMClientWrapper(self.config)

        ensure_dirs(self.config.chroma_db_dir)
        self.db_client = chromadb.PersistentClient(path=str(self.config.chroma_db_dir))

        # 默认使用 ONNX 嵌入函数
        self.emb_fn = ef.ONNXMiniLM_L6_V2()

        self.collection = self.db_client.get_or_create_collection(
            name="finance_docs", embedding_function=self.emb_fn
        )

    def build_index(self) -> Dict[str, int]:
        """构建索引"""
        stats = {"files_processed": 0, "chunks_added": 0}

        txt_files = list(self.config.docs_dir.glob("*.txt"))
        for txt_file in txt_files:
            if txt_file.name == "all_extracted.txt":
                continue

            content = txt_file.read_text(encoding="utf-8")
            cleaned = clean_text(content)
            chunks = split_text_by_sentence(cleaned, max_len=self.config.chunk_size)

            documents = []
            metadatas = []
            ids = []

            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({"source": txt_file.name, "chunk_index": i})
                ids.append(f"{txt_file.name}_{i}")

            if documents:
                self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
                stats["files_processed"] += 1
                stats["chunks_added"] += len(documents)

        return stats

    def query(self, question: str, top_k: int = 4) -> QueryResult:
        """执行查询"""
        results = self.collection.query(query_texts=[question], n_results=top_k)

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        context = "\n\n".join(docs)
        answer = self.llm.ask(question, context)

        return QueryResult(answer=answer, sources=metas, confidence=1.0)
