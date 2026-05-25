# classify_docs_bert.py - 一键运行，支持原始/微调模型切换
import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ===================== 配置：自由切换模型 =====================
USE_FINETUNED = True  # ← 改为 False 使用原始 hfl/chinese-bert-wwm-ext

FINETUNED_PATH = "./classify_model/best"
ORIGINAL_PATH = "hfl/chinese-bert-wwm-ext"
MODEL_PATH = FINETUNED_PATH if USE_FINETUNED else ORIGINAL_PATH

# 标签映射（与微调一致）
label_map = {0: "公司研究报告", 1: "行业周报", 2: "上市合规手册", 3: "合并财务报表"}

# ===================== 加载模型 =====================
print(f"正在加载模型：{'【微调模型】' if USE_FINETUNED else '【原始模型】'} {MODEL_PATH}")
print(f"设备：{'GPU' if torch.cuda.is_available() else 'CPU'}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print("模型加载成功！\n")

# ===================== 分类函数 =====================
def classify(text):
    inputs = tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    pred_id = probs.argmax().item()
    return {
        "type": label_map[pred_id],
        "confidence": round(float(probs.max().item()), 4),
        "all_scores": {label_map[i]: round(float(p), 4) for i, p in enumerate(probs)}
    }

# ===================== 自动读取 docs/ 下的所有 PDF 文本 =====================
DOCS_DIR = "docs"
results = []

if os.path.exists(DOCS_DIR):
    for file in os.listdir(DOCS_DIR):
        if file.endswith(".txt"):
            path = os.path.join(DOCS_DIR, file)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            result = classify(text)
            result["filename"] = file
            results.append(result)
            print(f"【{file}】 → {result['type']} | 置信度：{result['confidence']}")
else:
    print(f"目录 {DOCS_DIR} 不存在，请检查！")

# ===================== 保存结果 =====================
output = {
    "model_used": MODEL_PATH,
    "total_documents": len(results),
    "classification_results": results
}

with open("classification_report.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\n分类完成！详细报告已保存至 classification_report.json")