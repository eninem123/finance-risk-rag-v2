# extract_entities.py
# 银行级风控 RAG 系统 - Kimi 大模型版（可直接运行）
# 功能：规则+BER T双模式实体提取 + 风险评分 + Chroma向量库 + 循环问答
import json
import re
import os
import time
import shutil
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm  # 进度条
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

# 导入项目通用工具（已优化的文本处理、路径管理等）
from utils import (
    clean_text, load_json_file, save_json_file,
    ensure_dirs, normalize_path, calculate_risk_level
)

# ====================== 全局配置（适配你的项目结构）=======================
# 1. 路径配置（基于项目根目录，避免硬编码）
PROJECT_ROOT = normalize_path("")  # 项目根目录
DOCS_DIR = normalize_path("docs")  # 文档目录（存放待处理txt）
RULES_PATH = normalize_path("knowledge_base/risk_entities.json")  # 实体规则库
RAG_DB_DIR = normalize_path("rag_db")  # Chroma向量库目录
BERT_MODEL_PATH = normalize_path("bert_ner_model/best_model")  # 训练好的BERT模型路径
OUTPUT_JSON = normalize_path("docs/entities_extracted.json")  # 实体结果保存路径

# 2. Kimi大模型配置（替换为你的API密钥）
KIMI_API_KEY = "sk-VNPvMcWdMNXObfyi9fLMNSRBOYsmTgN420ugLlmV9z5RqxyE"  # 你的Kimi API Key
KIMI_BASE_URL = "https://api.moonshot.cn/v1"  # Kimi固定接口地址
KIMI_MODEL = "moonshot-v1-8k"  # Kimi支持的模型（避免404错误）

# 3. 实体提取配置
NUM_PATTERNS = {  # 金融数字实体正则（覆盖风控关键指标）
    "liquidity_risk": r'(现金储备|现金及现金等价物|cash.*reserve|流动性风险敞口).*?(\d+[,\d]*\.?\d*)\s*(亿|亿元|百万|million|billion|万)',
    "credit_rating": r'(评级|rating).*?(AAA|AA\+|AA|AA-|A\+|A|A-|BBB\+|BBB|BBB-|BB\+|BB|BB-)',
    "contingent_liability": r'(诉讼|pending litigation|或有负债).*?(\d+[,\d]*\.?\d*)\s*(亿|万元|USD|HKD)',
    "related_transaction": r'(关联交易金额|related party transaction).*?(\d+[,\d]*\.?\d*)\s*(亿|万元|HKD|USD)',
    "profit": r'(净利润|net profit|营收|revenue).*?(\d+[,\d]*\.?\d*)\s*(亿|亿元|万美元|USD)'
}

# ====================== 初始化核心组件 =======================
def init_kimi_client() -> OpenAI:
    """初始化Kimi大模型客户端（含错误处理）"""
    if not KIMI_API_KEY:
        raise ValueError("请设置Kimi API Key（KIMI_API_KEY变量）！")
    try:
        client = OpenAI(api_key=KIMI_API_KEY, base_url=KIMI_BASE_URL)
        # 测试客户端连通性
        client.models.list()
        print(f"✅ Kimi客户端初始化成功（模型：{KIMI_MODEL}）")
        return client
    except Exception as e:
        raise RuntimeError(f"Kimi客户端初始化失败：{str(e)}") from e

def init_chroma() -> chromadb.Collection:
    """初始化Chroma向量库（安全创建/删除，避免文件占用）"""
    # 1. 安全删除旧向量库
    safe_delete_rag_db()
    # 2. 确保向量库目录存在
    ensure_dirs(RAG_DB_DIR)
    # 3. 创建Chroma客户端和集合
    emb_fn = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
    chroma_client = chromadb.PersistentClient(path=RAG_DB_DIR)
    collection = chroma_client.create_collection(
        name="risk_entities",
        embedding_function=emb_fn,
        metadata={"description": "金融风控实体向量库"}
    )
    print(f"✅ Chroma向量库初始化成功（路径：{RAG_DB_DIR}）")
    return collection

def safe_delete_rag_db() -> None:
    """安全删除旧向量库（处理文件占用问题）"""
    if not os.path.exists(RAG_DB_DIR):
        return
    print(f"🔄 检测到旧向量库，尝试安全删除：{RAG_DB_DIR}")
    
    # 1. 先尝试Chroma内部删除
    try:
        chroma_client = chromadb.PersistentClient(path=RAG_DB_DIR)
        chroma_client.delete_collection("risk_entities")
        print("✅ Chroma集合已内部删除")
    except Exception as e:
        print(f"⚠️ Chroma内部删除失败（忽略）：{str(e)}")
    
    # 2. 强制删除目录（重试5次，处理文件占用）
    for retry in range(5):
        try:
            shutil.rmtree(RAG_DB_DIR)
            print(f"✅ 旧向量库目录已删除（重试{retry+1}次）")
            time.sleep(1)
            break
        except PermissionError:
            print(f"⚠️ 文件被占用，{2}秒后重试（{retry+1}/5）")
            time.sleep(2)
        except Exception as e:
            print(f"❌ 删除失败（{retry+1}/5）：{str(e)}")
            time.sleep(1)

# ====================== 实体提取核心逻辑 =======================
def load_risk_rules() -> Dict:
    """加载风险实体规则库（安全读取，支持默认规则）"""
    # 1. 尝试加载自定义规则库
    rules = load_json_file(RULES_PATH)
    if rules and "entities" in rules:
        print(f"✅ 加载自定义规则库成功（实体类型数：{len(rules['entities'])}）")
        return rules["entities"]
    
    # 2. 无自定义规则时，使用默认规则（避免运行失败）
    print(f"⚠️ 未找到自定义规则库，使用默认规则（路径：{RULES_PATH}）")
    default_rules = {
        "liquidity_risk": {
            "keywords": ["流动性风险", "现金储备不足", "流动性敞口"],
            "risk_score": 25,
            "description": "流动性风险：影响机构短期偿债能力的风险"
        },
        "credit_rating": {
            "keywords": ["信用评级下调", "AA+", "BBB-", "评级展望负面"],
            "risk_score": 20,
            "description": "信用评级：反映主体信用风险的评级结果"
        },
        "related_transaction": {
            "keywords": ["关联交易", "关联方资金占用", "非公允关联交易"],
            "risk_score": 15,
            "description": "关联交易：可能存在利益输送的交易行为"
        },
        "law_risk": {
            "keywords": ["诉讼", "行政处罚", "合规风险", "违反上市规则"],
            "risk_score": 30,
            "description": "法律合规风险：涉及诉讼、处罚的风险"
        }
    }
    return default_rules

def extract_rule_based_entities(text: str, rules: Dict) -> List[Dict]:
    """基于规则提取实体（含金融数字识别，去重+清洗）"""
    entities = []
    seen = set()  # 去重标记（避免重复提取同一实体）
    
    # 1. 清洗文本（统一标点、去除特殊字符）
    text = clean_text(text)
    if not text:
        return entities
    
    # 2. 关键词匹配提取（非数字实体）
    print(f"🔍 开始规则提取（文本长度：{len(text)}字符）")
    for ent_type, config in rules.items():
        for keyword in config["keywords"]:
            # 正则匹配关键词（不区分大小写，精准匹配单词边界）
            pattern = rf'\b{re.escape(keyword)}\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start = match.start()
                end = match.end()
                # 去重键（类型+关键词+位置，避免同一实体重复添加）
                dedup_key = f"{ent_type}_{keyword}_{start}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                
                # 提取上下文（前后各80字符，便于后续理解实体场景）
                context = text[max(0, start-80):end+80].replace("\n", " ").strip()
                
                entities.append({
                    "type": ent_type,
                    "text": keyword,
                    "start": start,
                    "end": end,
                    "context": context,
                    "confidence": 0.92,  # 规则提取置信度（固定高值）
                    "risk_score": config["risk_score"],
                    "description": config["description"]
                })
    
    # 3. 数字实体提取（如金额、评级）
    for ent_type, pattern in NUM_PATTERNS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
            # 解析匹配结果（确保分组存在）
            if len(match.groups()) < 2:
                continue
            metric_name = match.group(1).strip()  # 指标名称（如“流动性风险敞口”）
            amount = match.group(2).replace(",", "")  # 金额（去除千分位逗号）
            unit = match.group(3) if len(match.groups()) > 2 else ""  # 单位（如“亿元”）
            ent_text = f"{metric_name}{amount}{unit}"  # 完整实体文本
            start = match.start()
            end = match.end()
            
            # 去重
            dedup_key = f"{ent_type}_num_{start}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            
            entities.append({
                "type": ent_type,
                "text": ent_text,
                "start": start,
                "end": end,
                "context": match.group(0).replace("\n", " ").strip(),
                "confidence": 0.96,  # 数字提取置信度（更高）
                "risk_score": rules.get(ent_type, {}).get("risk_score", 15),  # 默认风险分
                "description": rules.get(ent_type, {}).get("description", f"金融数字指标：{metric_name}")
            })
    
    print(f"✅ 规则提取完成（实体数：{len(entities)}）")
    return entities

def extract_bert_entities(text: str) -> List[Dict]:
    """基于训练好的BERT模型提取实体（可选增强，失败不影响主流程）"""
    try:
        from transformers import (
            AutoModelForTokenClassification, 
            AutoTokenizer, 
            pipeline
        )
        # 1. 检查BERT模型路径
        if not os.path.exists(BERT_MODEL_PATH):
            raise FileNotFoundError(f"BERT模型路径不存在：{BERT_MODEL_PATH}")
        
        # 2. 加载BERT模型和分词器（适配你的训练结果）
        tokenizer = AutoTokenizer.from_pretrained(
            BERT_MODEL_PATH,
            local_files_only=True,
            model_max_length=512
        )
        model = AutoModelForTokenClassification.from_pretrained(
            BERT_MODEL_PATH,
            local_files_only=True,
            id2label={0:"O",1:"B-DATE",2:"I-DATE",3:"B-PER",4:"I-PER",
                      5:"B-ORG",6:"I-ORG",7:"B-MONEY",8:"I-MONEY",
                      9:"B-RISK",10:"I-RISK",11:"B-SEC",12:"I-SEC",
                      13:"B-REG",14:"I-REG",15:"B-LAW",16:"I-LAW"}  # 与训练时一致
        )
        
        # 3. 处理长文本（分片避免内存溢出）
        bert_entities = []
        chunk_size = 450  # 文本分片大小（适配512token）
        overlap = 50
        total_chunks = max(1, (len(text) + chunk_size - overlap - 1) // (chunk_size - overlap))
        
        print(f"🔍 开始BERT实体提取（分片数：{total_chunks}）")
        for i in tqdm(range(0, len(text), chunk_size - overlap), desc="BERT提取进度"):
            chunk = text[i:i+chunk_size]
            # 4. BERT实体提取（过滤低置信度）
            ner_pipe = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=-1  # 强制CPU（避免GPU内存不足）
            )
            results = ner_pipe(chunk)
            
            # 5. 处理BERT结果（补充分片位置偏移）
            for res in results:
                if res["score"] < 0.8:  # 过滤低置信度实体
                    continue
                # 修正实体在全文中的位置
                res["start"] += i
                res["end"] += i
                bert_entities.append({
                    "type": res["entity_group"],
                    "text": res["word"],
                    "start": res["start"],
                    "end": res["end"],
                    "context": text[max(0, res["start"]-60):res["end"]+60].replace("\n", " "),
                    "confidence": round(res["score"], 3),
                    "risk_score": 10,  # BERT实体默认风险分
                    "description": f"BERT自动识别：{res['entity_group']}类型实体"
                })
        
        print(f"✅ BERT提取完成（实体数：{len(bert_entities)}）")
        return bert_entities
    except Exception as e:
        print(f"⚠️ BERT实体提取跳过（原因：{str(e)}）")
        return []

def merge_entities(rule_ents: List[Dict], bert_ents: List[Dict]) -> List[Dict]:
    """合并规则提取和BERT提取的实体（去重，保留高置信度）"""
    merged = {}
    all_ents = rule_ents + bert_ents
    
    for ent in all_ents:
        # 去重键：实体类型+文本+起始位置（避免同一实体重复）
        dedup_key = f"{ent['type']}_{ent['text']}_{ent['start']}"
        # 保留置信度更高的实体
        if dedup_key not in merged or ent["confidence"] > merged[dedup_key]["confidence"]:
            merged[dedup_key] = ent
    
    final_ents = list(merged.values())
    print(f"✅ 实体合并去重完成（最终实体数：{len(final_ents)}）")
    return final_ents

# ====================== RAG向量库与问答 =======================
def build_rag_db(entities: List[Dict], collection: chromadb.Collection) -> None:
    """基于提取的实体构建Chroma向量库"""
    if not entities:
        print("⚠️ 无实体可构建向量库，跳过")
        return
    
    # 构建向量库所需数据（文档=实体描述+上下文，元数据=实体属性）
    docs = [
        f"【{ent['type']}】{ent['description']}\n实体内容：{ent['text']}\n上下文：{ent['context']}"
        for ent in entities
    ]
    metadatas = [
        {
            "type": ent["type"],
            "text": ent["text"],
            "risk_score": ent["risk_score"],
            "confidence": ent["confidence"],
            "start": ent["start"],
            "end": ent["end"]
        } for ent in entities
    ]
    ids = [f"ent_{i}" for i in range(len(entities))]
    
    # 批量添加到Chroma（处理可能的异常）
    try:
        collection.add(documents=docs, metadatas=metadatas, ids=ids)
        print(f"✅ RAG向量库构建完成（实体数：{len(entities)}，向量数：{len(docs)}）")
    except Exception as e:
        raise RuntimeError(f"向量库构建失败：{str(e)}") from e

def rag_qa(question: str, collection: chromadb.Collection, kimi_client: OpenAI) -> str:
    """RAG问答：检索向量库+Kimi生成回答"""
    # 1. 检索相关实体（Top4）
    print(f"\n🔍 检索相关实体（问题：{question[:50]}...）")
    try:
        query_res = collection.query(query_texts=[question], n_results=4)
        docs = query_res.get("documents", [[]])[0]
        metadatas = query_res.get("metadatas", [[]])[0]
        if not docs:
            return "未检索到与问题相关的实体信息。"
    except Exception as e:
        return f"向量库检索失败：{str(e)}"
    
    # 2. 构建上下文（格式化检索结果）
    context = "\n\n".join([
        f"【{meta['type']}】{meta['text']}\n风险分：{meta['risk_score']}\n上下文：{doc.split('上下文：')[-1].strip()}"
        for doc, meta in zip(docs, metadatas)
    ])
    
    # 3. 调用Kimi生成回答（金融风控场景prompt）
    prompt = f"""
你是专业金融风控顾问，基于以下实体信息回答问题，要求：
1. 严格引用上下文实体，不编造信息；
2. 回答简洁（<100字），重点突出风险点/关键指标；
3. 包含实体类型和风险评分（如有）。

上下文实体：
{context}

用户问题：{question}
"""
    try:
        response = kimi_client.chat.completions.create(
            model=KIMI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # 低温度保证回答稳定
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Kimi回答生成失败：{str(e)}"

# ====================== 主流程 =======================
def main():
    try:
        # 1. 初始化组件（Kimi+Chroma）
        print("="*50)
        print("🚀 开始金融风控实体提取与RAG问答流程")
        print("="*50)
        kimi_client = init_kimi_client()
        chroma_collection = init_chroma()
        
        # 2. 加载规则库
        print("\n" + "="*30)
        risk_rules = load_risk_rules()
        
        # 3. 读取待处理文本（docs目录下所有txt）
        print("\n" + "="*30)
        txt_files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".txt")]
        if not txt_files:
            raise FileNotFoundError(f"未在{DOCS_DIR}目录找到txt文件，请放入待处理文档！")
        # 读取第一个txt文件（如需处理多文件，可循环遍历）
        target_file = txt_files[0]
        file_path = normalize_path(os.path.join(DOCS_DIR, target_file))
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"✅ 读取文档成功（文件名：{target_file}，字符数：{len(text)}）")
        
        # 4. 多模式实体提取（规则+BERT）
        print("\n" + "="*30)
        rule_entities = extract_rule_based_entities(text, risk_rules)
        bert_entities = extract_bert_entities(text)
        final_entities = merge_entities(rule_entities, bert_entities)
        
        # 5. 计算风险等级
        total_risk = sum(ent["risk_score"] for ent in final_entities)
        risk_level = calculate_risk_level(total_risk)
        print(f"📊 风险评估结果：总风险分={total_risk} | 风险等级={risk_level}")
        
        # 6. 保存实体结果到JSON
        print("\n" + "="*30)
        result_data = {
            "extracted_at": datetime.now().isoformat(),
            "source_file": target_file,
            "total_entities": len(final_entities),
            "total_risk_score": total_risk,
            "risk_level": risk_level,
            "entities": final_entities
        }
        if save_json_file(result_data, OUTPUT_JSON):
            print(f"✅ 实体结果保存成功（路径：{OUTPUT_JSON}）")
        else:
            print(f"⚠️ 实体结果保存失败（路径：{OUTPUT_JSON}）")
        
        # 7. 构建RAG向量库
        print("\n" + "="*30)
        build_rag_db(final_entities, chroma_collection)
        
        # 8. 循环问答交互
        print("\n" + "="*50)
        print("💬 RAG风控问答系统已就绪（输入'exit'退出）")
        print("="*50)
        while True:
            question = input("\n请输入问题：").strip()
            if question.lower() in ["exit", "quit", "退出"]:
                print("👋 再见！")
                break
            if not question:
                continue
            answer = rag_qa(question, chroma_collection, kimi_client)
            print(f"\n📝 回答：{answer}")
    
    except Exception as e:
        print(f"\n❌ 流程运行失败：{str(e)}")
        exit(1)

if __name__ == "__main__":
    main()