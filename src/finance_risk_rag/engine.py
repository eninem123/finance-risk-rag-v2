"""
Finance-Risk-RAG Engine Module
==============================

Core RAG engine implementing vector storage and LLM integration.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import chromadb
from chromadb.utils import embedding_functions as ef
from openai import OpenAI

from .config import get_config
from .utils import clean_text, setup_logger, split_text_by_sentence


@dataclass
class QueryResult:
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMClientWrapper:
    """Wrapper for LLM interactions."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.client = None
        if self.config.llm_api_key:
            self.client = OpenAI(api_key=self.config.llm_api_key, base_url=self.config.llm_base_url)

    def call(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generic call to the LLM."""
        if not self.client:
            return "LLM API key not configured."

        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model_name, messages=messages, **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM Error: {str(e)}"


class RAGEngine:
    """Main RAG engine for indexing and querying."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = setup_logger("RAGEngine")
        self.llm = LLMClientWrapper(self.config)

        # Initialize ChromaDB
        self.db_client = chromadb.PersistentClient(path=str(self.config.chroma_db_dir))
        self.embedding_fn = ef.ONNXMiniLM_L6_V2()
        self.collection = self.db_client.get_or_create_collection(
            name="finance_risk_docs", embedding_function=self.embedding_fn
        )

    def index_document(self, text: str, source_name: str):
        """Index a document into the vector store."""
        cleaned = clean_text(text)
        chunks = split_text_by_sentence(cleaned, max_len=self.config.chunk_size)

        ids = [f"{source_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        self.logger.info(f"Indexed {len(chunks)} chunks from {source_name}")

    def build_index(self):
        """Index all text files in the docs directory."""
        txt_files = list(self.config.docs_dir.glob("*.txt"))
        for txt_file in txt_files:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()
            self.index_document(content, txt_file.name)

    def query(self, question: str, top_k: int = 4) -> QueryResult:
        """Perform a RAG query."""
        # Retrieve
        results = self.collection.query(query_texts=[question], n_results=top_k)

        context_parts = results.get("documents", [[]])[0]
        sources = results.get("metadatas", [[]])[0]

        context = "\n\n".join(context_parts)

        # Generate
        messages = [
            {
                "role": "system",
                "content": "You are a financial risk analyst. Answer based on the context.",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]

        answer = self.llm.call(messages)

        return QueryResult(answer=answer, sources=sources)
