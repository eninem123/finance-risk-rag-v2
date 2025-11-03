import os
import json
import torch
import numpy as np  # 确保导入numpy
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
            id2label=id2label,
            label2id=label2id
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"使用设备：{device}")
        print(f"✅ 模型加载成功：{MODEL_PATH}\n")
        return model, tokenizer, device
    except Exception as e:
        print(f"❌ 模型加载失败：{str(e)}")
        return None, None, None


# ====================== 实体聚合逻辑 =======================
def aggregate_entities(tokens, predictions, scores, tokenizer):
    entities = []
    current_entity = None
    for token, pred, score in zip(tokens, predictions, scores):
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue
        if pred.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            entity_type = pred[2:]
            current_entity = {
                "word": token.replace("##", ""),
                "entity_group": entity_type,
                "start": None,
                "end": None,
                "score": score
            }
        elif pred.startswith("I-") and current_entity:
            entity_type = pred[2:]
            if entity_type == current_entity["entity_group"]:
                current_entity["word"] += token.replace("##", "")
                current_entity["score"] = (current_entity["score"] + score) / 2
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    if current_entity:
        entities.append(current_entity)
    return entities


# ====================== 实体抽取逻辑（修复数据类型）=======================
def extract_entities_from_text(text: str, model, tokenizer, device, filename):
    if not model or not tokenizer or not text:
        return []
    
    total_chunks = max(1, (len(text) + CHUNK_SIZE - OVERLAP - 1) // (CHUNK_SIZE - OVERLAP))
    all_entities = []
    
    for i in tqdm(range(0, len(text), CHUNK_SIZE - OVERLAP), 
                  desc=f"处理 {filename} 分片", 
                  total=total_chunks, 
                  unit="片"):
        end = i + CHUNK_SIZE
        chunk = text[i:end]
        
        encoding = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        offset_mapping = encoding["offset_mapping"].squeeze(0).numpy()
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
        scores = torch.max(torch.softmax(logits, dim=2), dim=2).values.squeeze(0).cpu().numpy()
        
        pred_labels = [id2label[pred] for pred in predictions]
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())
        chunk_entities = aggregate_entities(tokens, pred_labels, scores, tokenizer)
        
        # 关键修正：将numpy.int64转为Python原生int
        for ent in chunk_entities:
            ent_start = None
            ent_end = None
            for idx, (start, end) in enumerate(offset_mapping):
                if start == 0 and end == 0:
                    continue
                token = tokens[idx].replace("##", "")
                if ent["word"].startswith(token) and ent_start is None:
                    ent_start = int(start + i)  # 转为Python int
                if ent["word"].endswith(token) and ent_end is None:
                    ent_end = int(end + i)    # 转为Python int
            if ent_start is not None and ent_end is not None:
                ent["start"] = ent_start
                ent["end"] = ent_end
                all_entities.append(ent)
    
    # 去重
    # 去重后添加过滤逻辑
    unique_entities = []
    seen = set()
    for ent in all_entities:
        if "start" not in ent or "end" not in ent:
            continue
        # 过滤规则：
        # 1. 过滤单个字的非金额/非日期实体（ORG/REG/LAW等单个字几乎都是误判）
        entity_type = ent["entity_group"]
        entity_word = ent["word"]
        if len(entity_word) == 1 and entity_type not in ["MONEY", "DATE"]:
            continue
        # 2. 过滤置信度低于0.5的实体（低置信度大概率是误判）
        if ent["score"] < 0.5:
            continue
        # 3. 去重
        key = (entity_word, ent["start"], ent["end"], entity_type)
        if key not in seen:
            seen.add(key)
            unique_entities.append(ent)
    
    return unique_entities


# ====================== 主函数 =======================
def main():
    model, tokenizer, device = load_model_tokenizer()
    if not model or not tokenizer:
        return
    
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"⚠️ 已创建文档目录：{DOCS_DIR}，请放入txt文件后重试")
        return
    
    txt_files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".txt")]
    if not txt_files:
        print(f"⚠️ 未在{DOCS_DIR}找到txt文件")
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
                    "实体内容": ent["word"],
                    "实体类型": ent["entity_group"],
                    "起始位置": ent["start"],
                    "结束位置": ent["end"],
                    "置信度": round(ent["score"], 4)
                } for ent in entities
            ]
            results[filename] = formatted_entities
            print(f"✅ {filename} 处理完成，抽取到 {len(formatted_entities)} 个实体")
        except Exception as e:
            print(f"❌ {filename} 处理失败：{str(e)}")
    
    # 保存结果（现在所有数据都是Python原生类型，可正常序列化）
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n🎉 所有文件处理完成，结果保存至：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()