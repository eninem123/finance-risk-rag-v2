# bert_finetune.py
from utils import clean_text, setup_logger, ensure_dirs
import torch
import numpy as np
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    get_linear_schedule_with_warmup  # 正确名称
)
from tqdm import tqdm
import os

# 初始化日志
logger = setup_logger(name="bert_finetune", log_file="bert_train.log", level=logging.INFO)

# ====================== 配置 =======================
class Config:
    train_path = "dataset/train/ner_train.txt"
    dev_path = "dataset/dev/ner_dev.txt"
    label_list = [
        "O", "B-DATE", "I-DATE", "B-PER", "I-PER",
        "B-ORG", "I-ORG", "B-MONEY", "I-MONEY",
        "B-RISK", "I-RISK", "B-SEC", "I-SEC",
        "B-REG", "I-REG", "B-LAW", "I-LAW"
    ]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    model_name = "hfl/chinese-bert-wwm-ext"
    max_seq_len = 512
    batch_size = 4
    epochs = 5
    learning_rate = 2e-5
    save_dir = "bert_ner_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# ====================== 数据集 =======================
class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        self.data = self.load_conll_data(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.label2id = config.label2id

    def load_conll_data(self, data_path):
        data = []
        current_tokens = []
        current_labels = []
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        if current_tokens:
                            data.append({"tokens": current_tokens, "labels": current_labels})
                            current_tokens = []
                            current_labels = []
                        continue
                    parts = line.split()
                    if len(parts) != 2:
                        logger.warning(f"第{line_num}行格式错误，跳过 → {line}")
                        continue
                    token, label = parts[0], parts[1]
                    token = clean_text(token)
                    if token:
                        current_tokens.append(token)
                        current_labels.append(label)
            if current_tokens:
                data.append({"tokens": current_tokens, "labels": current_labels})
            print(f"成功加载{len(data)}条句子（来自{data_path}）")
            return data
        except FileNotFoundError:
            print(f"错误：未找到文件 {data_path}")
            exit(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        labels = item["labels"]
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=config.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            else:
                label = labels[word_id] if word_id < len(labels) else "O"
                aligned_labels.append(self.label2id[label])
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }

# ====================== 模型 =======================
def init_model_tokenizer():
    tokenizer = BertTokenizerFast.from_pretrained(config.model_name)
    model = BertForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(config.label_list),
        id2label=config.id2label,
        label2id=config.label2id
    )
    model.to(config.device)
    return model, tokenizer

def train_epoch(model, dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=2).cpu().numpy()
        labels_np = labels.cpu().numpy()
        for p, l in zip(preds, labels_np):
            mask = l != -100
            all_preds.extend(p[mask])
            all_labels.extend(l[mask])
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, precision, recall, f1

def eval_epoch(model, dataloader):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(logits, dim=2).cpu().numpy()
            labels_np = labels.cpu().numpy()
            for p, l in zip(preds, labels_np):
                mask = l != -100
                all_preds.extend(p[mask])
                all_labels.extend(l[mask])
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, precision, recall, f1

# ====================== 主函数 =======================
def main():
    model, tokenizer = init_model_tokenizer()
    logger.info(f"模型加载完成，使用设备：{config.device}")
    ensure_dirs(config.save_dir)
    train_dataset = NERDataset(config.train_path, tokenizer, config.max_seq_len)
    dev_dataset = NERDataset(config.dev_path, tokenizer, config.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)
    print(f"数据加载完成：训练集{len(train_dataset)}条，验证集{len(dev_dataset)}条")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, eps=1e-8)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_f1 = 0.0
    os.makedirs(config.save_dir, exist_ok=True)
    print(f"\n开始训练（共{config.epochs}轮）：")

    for epoch in range(config.epochs):
        print(f"\n===== Epoch {epoch+1}/{config.epochs} =====")
        train_loss, train_acc, train_p, train_r, train_f1 = train_epoch(model, train_loader, optimizer, scheduler)
        dev_loss, dev_acc, dev_p, dev_r, dev_f1 = eval_epoch(model, dev_loader)
        print(f"训练集：损失={train_loss:.4f} | 准确率={train_acc:.4f} | F1={train_f1:.4f}")
        print(f"验证集：损失={dev_loss:.4f} | 准确率={dev_acc:.4f} | F1={dev_f1:.4f}")
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            model.save_pretrained(os.path.join(config.save_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(config.save_dir, "best_model"))
            print(f"最佳模型已保存至：{os.path.join(config.save_dir, 'best_model')}")

    print(f"\n训练完成！最佳验证集F1：{best_f1:.4f}")

if __name__ == "__main__":
    main()