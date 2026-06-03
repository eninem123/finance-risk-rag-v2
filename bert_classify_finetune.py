# bert_classify_finetune.py  微调 BERT 分类模型
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import json

# 1. 加载数据
class ClassifyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

# 2. 准备数据
texts = [
    "小鹏汽车2024年现金储备1507亿元，同比增长12%",
    "平安证券新能源汽车行业周报：小鹏汽车发布二季度财报",
    "汉坤律所发布《履行持续义务：主要证券交易所上市公司手册》",
    "SQM 智利化工合并财务报表（截至2017年12月31日）"
]
labels = [0, 1, 2, 3]
label_map = {0: "公司研究报告", 1: "行业周报", 2: "上市合规手册", 3: "合并财务报表"}

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
model = AutoModelForSequenceClassification.from_pretrained(
    "hfl/chinese-bert-wwm-ext", num_labels=4
)

train_dataset = ClassifyDataset(texts, labels, tokenizer)

# 3. 训练
training_args = TrainingArguments(
    output_dir="./classify_model",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    save_steps=10,
    logging_steps=1,
    learning_rate=5e-5,
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()

# 4. 保存
model.save_pretrained("./classify_model/best")
tokenizer.save_pretrained("./classify_model/best")
print("分类模型微调完成！")