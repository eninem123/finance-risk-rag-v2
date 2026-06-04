# bert_finetune.py
import logging
import os

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BertForTokenClassification,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from utils import ensure_dirs, setup_logger

# 初始化日志
logger = setup_logger(name="bert_finetune", log_file="bert_train.log", level=logging.INFO)


# ====================== 配置 =======================
class Config:
    train_path = "dataset/train/ner_train.txt"
    dev_path = "dataset/dev/ner_dev.txt"
    label_list = [
        "O",
        "B-DATE",
        "I-DATE",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-RISK",
        "I-RISK",
    ]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    model_name = "hfl/chinese-bert-wwm-ext"
    max_len = 128
    batch_size = 16
    epochs = 5
    lr = 2e-5
    save_path = "hfl/chinese-bert-wwm-ext-finetuned"


# ====================== 数据集 = :Dataset =======================
class FinanceNERDataset(Dataset):
    def __init__(self, path, tokenizer, label2id, max_len):
        self.data = self._load_data(path)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def _load_data(self, path):
        if not os.path.exists(path):
            logger.warning(f"数据路径不存在: {path}")
            return []
        items = []
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n\n")
            for line in lines:
                words, labels = [], []
                for char_line in line.split("\n"):
                    parts = char_line.split()
                    if len(parts) == 2:
                        words.append(parts[0])
                        labels.append(parts[1])
                if words:
                    items.append({"words": words, "labels": labels})
        return items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["words"],
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        labels = [self.label2id.get(label, 0) for label in item["labels"]]
        # 填充标签
        labels = labels[: self.max_len] + [0] * (self.max_len - len(labels))

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ====================== 训练 & 评估 =======================
def train():
    config = Config()
    ensure_dirs(config.save_path)

    tokenizer = BertTokenizerFast.from_pretrained(config.model_name)
    model = BertForTokenClassification.from_pretrained(
        config.model_name, num_labels=config.num_labels
    )

    train_dataset = FinanceNERDataset(config.train_path, tokenizer, config.label2id, config.max_len)
    dev_dataset = FinanceNERDataset(config.dev_path, tokenizer, config.label2id, config.max_len)

    if not train_dataset:
        logger.error("训练集为空，请检查路径。")
        return

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")
        evaluate(model, dev_loader, device, config.id2label)

    # 保存模型
    model.save_pretrained(config.save_path)
    tokenizer.save_pretrained(config.save_path)
    logger.info(f"模型已保存至: {config.save_path}")


def evaluate(model, loader, device, id2label):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=2)

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # 过滤掉填充部分 (假设 0 是 O 标签且不计入核心评估，或根据实际需要调整)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(all_labels, all_preds)
    logger.info(
        f"Eval - Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )


if __name__ == "__main__":
    train()
