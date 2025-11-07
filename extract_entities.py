# extract_entities.py
# 银行级风控 RAG 系统 - 完整稳定版（已修复 token 超限 + 分块 + 融合 + 防 TPM）
import json
import re
import os
from datetime import datetime
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from config import LLM_API_KEY, LLM_BASE_URL
from utils import (
    clean_text, load_stopwords, calculate_risk_level,
    safe_delete_rag_db, setup_logger
)
from extract_entities_bert import extract_entities_from_text, load_model_tokenizer
import time
from utils import ensure_dirs  # 确保目录存在
# 在初始化 logger 前调用
ensure_dirs("logs")  # 自动创建 logs 目录
logger = setup_logger("extract_entities", "logs/extract_entities.log")
# ====================== 初始化 ======================

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

# 安全删除旧库
safe_delete_rag_db()

# 初始化 Chroma
embedding_function = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
chroma_client = chromadb.PersistentClient(path="rag_db")
collection = chroma_client.create_collection(
    name="risk_entities",
    embedding_function=embedding_function
)

# ====================== 规则库 ======================
def load_entity_rules():
    with open("knowledge_base/risk_entities.json", "r", encoding="utf-8") as f:
        return json.load(f)

# ====================== 规则提取 ======================
def extract_entities_rule_based(text, rules):
    entities = []
    seen = set()

    num_patterns = {
        "liquidity_risk": r'(现金储备|现金及现金等价物|cash.*reserve).*?(\d+[,\d]*\.?\d*)\s*(亿|亿元|百万|million|billion)',
        "credit_rating": r'(评级|rating).*?(AAA|AA\+|AA|AA-|A\+|A|A-|BBB\+|BBB|BBB-)',
        "contingent_liability": r'(诉讼|pending litigation).*?(\d+[,\d]*\.?\d*)\s*(亿|万元|USD)',
        "related_transaction": r'(关联交易金额|related party).*?(\d+[,\d]*\.?\d*)\s*(亿|万元|HKD|USD)'
    }

    for entity_type, config in rules.items():
        for keyword in config["keywords"]:
            pattern = rf'\b{re.escape(keyword)}\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start = match.start()
                key = f"{entity_type}_{keyword}_{start}"
                if key in seen: continue
                seen.add(key)
                context = text[max(0, start-80):start+len(keyword)+80].replace("\n", " ").strip()
                entities.append({
                    "type": entity_type,
                    "text": keyword,
                    "context": context,
                    "source": "rule",
                    "risk_score": config.get("risk_score", 10),
                    "confidence": 1.0
                })
    return entities

# ====================== 文本分块 ======================
def chunk_text(text, chunk_size=1000, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
        if i >= len(words) and len(chunks) > 1:
            break
    return chunks

# ====================== 构建 RAG 向量库 ======================
def build_rag_vector_db(entities):
    texts = []
    metadatas = []
    ids = []

    for i, ent in enumerate(entities):
        content = f"{ent['type']} | {ent['text']} | 风险分: {ent['risk_score']} | 置信: {ent['confidence']:.2f}"
        if 'context' in ent:
            content += f" | 上下文: {ent['context'][:200]}"
        texts.append(content)
        metadatas.append({
            "type": ent['type'],
            "text": ent['text'],
            "risk_score": ent['risk_score'],
            "confidence": ent['confidence'],
            "source": ent.get('source', 'bert')
        })
        ids.append(f"ent_{i}")

    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    logger.info(f"RAG 向量库构建完成，共 {len(texts)} 条记录")

# ====================== RAG 问答 ======================
def rag_query(question):
    try:
        results = collection.query(
            query_texts=[question],
            n_results=5
        )
        context = "\n".join([
            f"【{m['type']}】{m['text']} (风险分: {m['risk_score']}, 置信: {m['confidence']:.2f})"
            for m in results['metadatas'][0]
        ]) if results['metadatas'] else "未检索到相关实体。"

        prompt = f"""
你是一个银行风控专家。基于以下检索到的风险实体，简洁、专业地回答问题。

【检索到的风险实体】：
{context}

【用户问题】：
{question}

【要求】：
- 回答简洁、专业、数据化
- 若无数据，说“未发现相关风险”
- 输出纯文本，无 JSON

回答：
"""
        time.sleep(1)  # 防 TPM 超限
        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"RAG 查询失败: {e}")
        return f"问答失败：{e}"

# ====================== 融合实体 ======================
def merge_entities(rule_entities, bert_entities):
    merged = {}
    for ent in rule_entities + bert_entities:
        key = (ent['text'], ent['type'])
        if key not in merged:
            merged[key] = ent
        else:
            # 融合置信度和风险分
            old = merged[key]
            old['confidence'] = max(old['confidence'], ent['confidence'])
            old['risk_score'] = max(old['risk_score'], ent.get('risk_score', 10))
    return list(merged.values())

# ====================== 主流程 ======================
if __name__ == "__main__":
    print("开始实体提取与 RAG 构建...")
    logger.info("=== 开始实体提取 ===")

    # 加载模型
    model, tokenizer, device = load_model_tokenizer()
    if not model:
        print("BERT 模型加载失败，仅使用规则提取")
        bert_entities = []
    else:
        bert_entities = []

    # 加载规则
    try:
        rules = load_entity_rules()
    except Exception as e:
        logger.error(f"规则加载失败: {e}")
        rules = {}

    # 读取全量文本
    txt_path = "docs/all_extracted.txt"
    if not os.path.exists(txt_path):
        print(f"错误：未找到 {txt_path}，请先运行 extract_text.py")
        exit(1)

    with open(txt_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    full_text = clean_text(full_text)
    logger.info(f"文本总长度: {len(full_text)} 字符")

    # 分块处理 BERT（避免 OOM）
    if model:
        print("正在分块运行 BERT NER...")
        chunks = chunk_text(full_text, chunk_size=400, overlap=50)
        bert_entities = []
        for i, chunk in enumerate(chunks):
            print(f"  处理分块 {i+1}/{len(chunks)}")
            try:
                # 关键：接收返回的结构化实体
                ents = extract_entities_from_text(chunk, model, tokenizer, device, f"chunk_{i}")
                bert_entities.extend(ents)
                print(f"    → 本块提取 {len(ents)} 个实体")
                time.sleep(0.1)
            except Exception as e:
                logger.warning(f"分块 {i} BERT 失败: {e}")
    else:
    bert_entities = []

    # 规则提取（全局）
    print("正在运行规则提取...")
    rule_entities = extract_entities_rule_based(full_text, rules)

    # 融合
    final_entities = merge_entities(rule_entities, bert_entities)
    logger.info(f"规则实体: {len(rule_entities)}，BERT 实体: {len(bert_entities)}，融合后: {len(final_entities)}")

    # 计算风险
    total_risk = sum(e["risk_score"] for e in final_entities)
    risk_level = calculate_risk_level(total_risk)

    # 保存结果
    result = {
        "extracted_at": datetime.now().isoformat(),
        "total_entities": len(final_entities),
        "total_risk_score": total_risk,
        "risk_level": risk_level,
        "entities": [
            {
                "type": e["type"],
                "text": e["text"],
                "risk_score": e["risk_score"],
                "confidence": round(e["confidence"], 4),
                "source": e.get("source", "bert")
            }
            for e in final_entities
        ]
    }
    os.makedirs("docs", exist_ok=True)
    with open("docs/entities_extracted.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 构建 RAG
    build_rag_vector_db(final_entities)

    print(f"\n实体提取完成！")
    print(f"   实体数: {len(final_entities)}")
    print(f"   总风险: {total_risk}/100 ({risk_level})")
    print(f"   保存: docs/entities_extracted.json")

    print("\nTop 5 高风险实体：")
    top5 = sorted(final_entities, key=lambda x: x["risk_score"], reverse=True)[:5]
    for e in top5:
        print(f"   {e['type']:20} | {e['text']:30} | 分数: {e['risk_score']:2} | 置信: {e['confidence']:.2f}")

    # ====================== 循环问答 ======================
    print("\nRAG 风控问答系统已就绪！输入 'exit' 退出。")
    while True:
        try:
            question = input("\n问：").strip()
            if question.lower() in ["exit", "quit", "退出"]:
                print("再见！")
                break
            if not question:
                continue
            answer = rag_query(question)
            print(f"答：{answer}")
        except KeyboardInterrupt:
            print("\n再见！")
            break