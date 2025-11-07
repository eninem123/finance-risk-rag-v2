# extract_entities_bert.py
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict
from tqdm import tqdm

# ====================== 配置参数 =======================
MODEL_PATH = "bert_ner_model/best_model"
DOCS_DIR = "docs"
OUTPUT_FILE = os.path.join(DOCS_DIR, "entities_extracted.json")
LABEL_LIST = [
    "O", "B-DATE", "I-DATE", "B-PER", "I-PER",
    "B-ORG", "I-ORG", "B-MONEY", "I-MONEY",
    "B-RISK", "I-RISK", "B-SEC", "I-SEC",
    "B-REG", "I-REG", "B-LAW", "I-LAW"
]
id2label = {i: label for i, label in enumerate(LABEL_LIST)}
label2id = {v: k for k, v in id2label.items()}
MAX_SEQ_LEN = 512
CHUNK_SIZE = 450
OVERLAP = 50


# ====================== 加载模型和分词器 =======================
def load_model_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            model_max_length=MAX_SEQ_LEN
        )
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            num_labels=len(LABEL_LIST),
            id2label=id2label,
            label2id=label2id
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"使用设备：{device}")
        print(f"模型加载成功：{MODEL_PATH}\n")
        return model, tokenizer, device
    except Exception as e:
        print(f"模型加载失败：{str(e)}")
        return None, None, None


# ====================== 风险分映射 =======================
def get_risk_score_by_type(entity_type):
    mapping = {
        "ORG": 10,
        "MONEY": 25,
        "DATE": 5,
        "RISK": 30,
        "SEC": 20,
        "REG": 15,
        "LAW": 18,
        "PER": 8
    }
    return mapping.get(entity_type, 10)


# ====================== 核心 NER 推理 =======================
def extract_entities_from_text(text, model, tokenizer, device, filename="unknown"):
    if not text.strip():
        return []

    # 1. 分词 + 获取 offset
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=True,
        return_offsets_mapping=True
    )
    offset_mapping = encoded["offset_mapping"][0].cpu().numpy()

    # 2. 构造 model 输入（移除 offset_mapping）
    inputs = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }
    if "token_type_ids" in encoded:
        inputs["token_type_ids"] = encoded["token_type_ids"].to(device)

    # 3. 推理
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
        scores = F.softmax(outputs.logits[0], dim=1).cpu().numpy()

    # 4. 解析实体
    entities = []
    current = None
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

    for i, (token, pred_id, (start, end)) in enumerate(zip(tokens, predictions, offset_mapping)):
        if token in ["[CLS]", "[SEP]", "[PAD]"] or start == end == 0:
            continue

        label = id2label[pred_id]
        score = scores[i][pred_id]

        if label.startswith("B-"):
            if current:
                entities.append(current)
            current = {
                "word": text[start:end],
                "entity_group": label[2:],
                "score": score,
                "start": int(start),
                "end": int(end)
            }
        elif label.startswith("I-") and current and label[2:] == current["entity_group"]:
            current["word"] += text[current["end"]:end]
            current["end"] = int(end)
            current["score"] = max(current["score"], score)

    if current:
        entities.append(current)

    # 5. 后处理 + 过滤
    filtered = []
    seen = set()
    for ent in entities:
        word = ent["word"].strip()
        typ = ent["entity_group"]
        if ent["score"] < 0.5 or len(word) == 0:
            continue
        if len(word) == 1 and typ not in ["MONEY", "DATE"]:
            continue
        key = (word, typ)
        if key not in seen:
            seen.add(key)
            filtered.append({
                "type": typ,
                "text": word,
                "risk_score": get_risk_score_by_type(typ),
                "confidence": round(float(ent["score"]), 4),
                "source": "bert"
            })
    return filtered


# ====================== 主函数 =======================
def main():
    model, tokenizer, device = load_model_tokenizer()
    if not model or not tokenizer:
        return

    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"已创建文档目录：{DOCS_DIR}，请放入txt文件后重试")
        return

    txt_files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".txt")]
    if not txt_files:
        print(f"未在{DOCS_DIR}找到txt文件")
        return

    results = {}
    for filename in tqdm(txt_files, desc="总体进度", total=len(txt_files), unit="文件"):
        file_path = os.path.join(DOCS_DIR, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            print(f"\n开始处理：{filename}（长度：{len(text)}字符）")

            entities = extract_entities_from_text(text, model, tokenizer, device, filename)
            formatted_entities = [
                {
                    "实体内容": ent["text"],
                    "实体类型": ent["type"],
                    "起始位置": None,
                    "结束位置": None,
                    "置信度": ent["confidence"]
                } for ent in entities
            ]
            results[filename] = formatted_entities
            print(f"{filename} 处理完成，抽取到 {len(formatted_entities)} 个实体")
        except Exception as e:
            print(f"{filename} 处理失败：{str(e)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n所有文件处理完成，结果保存至：{OUTPUT_FILE}")


# ====================== 测试入口 =======================
if __name__ == "__main__":
    model, tokenizer, device = load_model_tokenizer()
    if model:
        test_text = "小鹏汽车2024年现金储备为1507亿元，存在信用风险。"
        ents = extract_entities_from_text(test_text, model, tokenizer, device, "test")
        print("BERT 测试输出：", ents)