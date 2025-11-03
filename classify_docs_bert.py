import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer
import numpy as np

from transformers import AutoTokenizer, AutoModel

MODEL_PATH = os.path.join("D:\\finance-risk-rag", "hfl", "chinese-bert-wwm-ext")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH)

# 预定义文档类型（银行风控场景常用）
DOC_TYPES = [
    "年度财务报告", "季度财务报告", "风险评估报告", 
    "信贷审批报告", "贷后检查报告", "监管合规报告", "其他"
]

class BERTClassifier:
    def __init__(self):
        # 加载轻量级BERT模型（平衡速度和精度）
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-chinese',
            num_labels=len(DOC_TYPES)
        )
        # 加载句子嵌入模型（用于文本截断处理）
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()  # 推理模式

    def _truncate_text(self, text, max_tokens=512):
        """截断文本到BERT最大输入长度（避免显存爆炸）"""
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) <= max_tokens:
            return text
        # 取最前面和最后面的内容（保留关键信息）
        truncated_tokens = tokens[:300] + tokens[-212:]
        return self.tokenizer.convert_tokens_to_string(truncated_tokens)

    def classify_document(self, text):
        """
        分类文档类型
        :param text: 文档文本内容
        :return: 分类结果（类型+置信度）
        """
        if not text.strip():
            return {"type": "其他", "confidence": 1.0}
        
        # 预处理文本
        truncated_text = self._truncate_text(text)
        inputs = self.tokenizer(
            truncated_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # 模型推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # 解析结果
        max_idx = np.argmax(probabilities)
        return {
            "type": DOC_TYPES[max_idx],
            "confidence": float(probabilities[max_idx])
        }

if __name__ == "__main__":
    # 测试代码
    classifier = BERTClassifier()
    sample_text = "本报告旨在评估某企业2024年度的信贷风险，包括其偿债能力、现金流状况..."
    print(classifier.classify_document(sample_text))