# rag_core.py 向量化 + Chroma 持久化 + LLM API 调用 智能问答
import os
import json
import math
from typing import List
from pathlib import Path
import config
# 新增：兼容 huggingface_hub 0.19.0+（替换被移除的 cached_download）
from huggingface_hub import hf_hub_download
import huggingface_hub
huggingface_hub.cached_download = hf_hub_download  # 替换被移除的函数

# Chroma 和 Embedding
import chromadb
from chromadb.utils import embedding_functions as ef

# 备用：sentence-transformers
from sentence_transformers import SentenceTransformer

# LLM 客户端（Moonshot/OpenAI 使用相同 OpenAI-compatible SDK）
from openai import OpenAI
import time
# rag_core.py 头部导入
from utils import clean_text, split_text_by_sentence, ensure_dirs

# 猴子补丁：用 hf_hub_download 替代 cached_download（两者功能兼容）
huggingface_hub.cached_download = hf_hub_download
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_DIR = config.CHROMA_DB_DIR

# 初始化 LLM 客户端（从环境变量读取密钥）
API_KEY = config.LLM_API_KEY
BASE_URL = config.LLM_BASE_URL

if not API_KEY:
    print("警告：未检测到 LLM API key（环境变量 OPENAI_API_KEY 或 MOONSHOT_API_KEY）。请设置后再调用 LLM 功能。")

def init_llm_client():
    # 使用 openai python 客户端（兼容 Moonshot）
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        return client
    except Exception as e:
        print("初始化 LLM client 失败：", e)
        return None

llm_client = init_llm_client()

def ask_llm(query: str, context: str, client=llm_client, model_name: str = "moonshot-v1-8k"):
    if client is None:
        raise RuntimeError("LLM client 未初始化，请设置 OPENAI_API_KEY / MOONSHOT_API_KEY")
    # 组合 prompt
    system_prompt = "你是一名金融风险分析顾问，回答时引用上下文并给出简明结论。"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"参考以下上下文来回答问题：\n\n{context}\n\n问题：{query}"}
    ]
    # 调用 chat completions
    try:
        resp = client.chat.completions.create(model=model_name, messages=messages, temperature=0.0, max_tokens=512)
        # 兼容不同 SDK 返回结构
        try:
            return resp.choices[0].message.content
        except Exception:
            # fallback
            return resp.choices[0].text
    except Exception as e:
        print("LLM 调用失败：", e)
        raise

# 尝试使用 ONNX MiniLM 嵌入（Chroma 提供的包装）
def get_embedding_function():
    try:
        # preferred_providers 可用值 ["CPUExecutionProvider"] 等
        emb_fn = ef.ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
        print("使用 ONNXMiniLM_L6_V2 作为嵌入函数")
        return emb_fn
    except Exception as e:
        print("ONNXMiniLM_L6_V2 不可用，fallback 到 sentence-transformers。错误：", e)
        # fallback
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        def sbert_embed(texts: List[str]):
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()
        return sbert_embed

# 修改 chunk_text 函数（用按句子拆分替代固定长度拆分）
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    # 1. 清洗文本（去除特殊字符，统一标点）
    text = clean_text(text)
    # 2. 按句子拆分（保留语义完整性）
    sentences = split_text_by_sentence(text, max_len=chunk_size)
    # 3. 处理重叠（避免句子断裂）
    chunks = []
    for i in range(len(sentences)):
        if i == 0:
            current = sentences[i]
        else:
            # 重叠前一个句子的最后 overlap 字符（确保上下文连贯）
            prev_sent = sentences[i-1]
            overlap_part = prev_sent[-overlap:] if len(prev_sent) >= overlap else prev_sent
            current = overlap_part + sentences[i]
        # 确保单chunk不超过最大长度
        if len(current) > chunk_size:
            current = current[:chunk_size]
        chunks.append(current)
    return chunks

# 优化 build_db 函数：确保目录存在（避免路径错误）
def build_db():
    # 新增：确保 Chroma 目录存在
    ensure_dirs(CHROMA_DIR)
    emb_fn = get_embedding_function()
    # chroma 持久化 client
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection_name = "finance_docs"
    # 尝试创建或获取集合
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name, embedding_function=emb_fn)
    # 遍历 docs/*.txt
    for fname in os.listdir(DOCS_DIR):
        if not fname.lower().endswith(".txt"):
            continue
        fpath = os.path.join(DOCS_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as fr:
            txt = fr.read()
        chunks = chunk_text(txt, chunk_size=800, overlap=100)
        ids = [f"{fname}__{i}" for i in range(len(chunks))]
        metadatas = [{"source": fname, "chunk_index": i} for i in range(len(chunks))]
        # 如果使用 emb_fn 是函数则传 embeddings param；如果是 chroma 的 embedding object，直接交给 collection.add
        try:
            # 若 emb_fn 是一个包装器（带 .get_embedding），尝试直接使用 collection.add
            collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            print(f"已添加到 Chroma: {fname}，chunks={len(chunks)}")
        except Exception as e:
            # fallback：显式先计算向量
            try:
                if callable(emb_fn):
                    vectors = emb_fn(chunks)
                else:
                    vectors = [emb_fn.get_embedding(c) for c in chunks]
                collection.add(
                    documents=chunks,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=vectors
                )
                print(f"(fallback) 已添加带向量到 Chroma: {fname}")
            except Exception as e2:
                print("将文档写入 Chroma 失败：", e, e2)

def query(query_text: str, top_k: int = 4):
    # 载入 chroma 并检索
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name="finance_docs")
    # 使用 collection.query
    try:
        qres = collection.query(query_texts=[query_text], n_results=top_k)
        # qres format: {"ids": [[...]], "distances": [[...]], "documents":[[...]...], "metadatas":[[...]...]}
        docs = qres.get("documents", [[]])[0]
        metadatas = qres.get("metadatas", [[]])[0]
    except Exception as e:
        print("Chroma 查询失败：", e)
        return None

    # 组合 context（把若干片段拼接）
    context = "\n\n".join(docs)
    # 交给 LLM
    try:
        answer = ask_llm(query_text, context)
        return {"answer": answer, "sources": metadatas}
    except Exception as e:
        print("LLM 回答失败：", e)
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-db", action="store_true", help="构建 Chroma 向量库")
    parser.add_argument("--query", type=str, help="向量检索+LLM问答")
    args = parser.parse_args()
    if args.build_db:
        build_db()
    if args.query:
        r = query(args.query)
        print("Answer:", r)
# 测试
# 构建向量库（会遍历 docs/*.txt）
# python rag_core.py --build-db

# 测试查询（先构建好数据库）
# python rag_core.py --query "这份报告有哪些风险点？"