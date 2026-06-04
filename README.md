# Finance-Risk-RAG v2.1

<div align="center">

**银行级多语言财务文本风控AI系统**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

[功能特性](#-功能特性) • [快速开始](#-快速开始) • [架构设计](#-架构设计) • [命令行工具](#-命令行工具) • [贡献指南](#-贡献指南)

</div>

---

## 项目简介

Finance-Risk-RAG 是一套针对财务文档的智能风控系统。它通过 OCR 识别、文档分类、风险实体抽取和 RAG (检索增强生成) 技术，实现金融风险的自动化分析与预警。

### 核心能力

| 能力 | 实现方案 |
|------|---------|
| **智能 OCR** | Tesseract 5.5 + 图像增强，支持表格与复杂布局 |
| **文档分类** | 基于 LLM 的自动文档属性识别 |
| **风险提取** | 规则引擎 + BERT 命名实体识别 |
| **智能问答** | 基于 ChromaDB 与 ONNX 嵌入模型的 RAG 系统 |

---

## 🏗️ 架构设计

项目采用标准 Python 包结构，核心逻辑位于 `src/finance_risk_rag`：

```text
finance-risk-rag/
├── src/finance_risk_rag/
│   ├── config.py          # 配置管理
│   ├── models.py          # 数据模型 (Entity, QueryResult)
│   ├── utils.py           # 工具函数 (文本清洗, 风险计算)
│   ├── processor.py       # 文档处理 (OCR, 分类)
│   ├── extractor.py       # 实体提取 (规则, BERT)
│   ├── engine.py          # RAG 引擎 (向量检索)
│   └── llm.py             # LLM 客户端封装
├── main.py                # 统一命令行入口
├── tests/                 # 单元测试 suite
├── scripts/               # 训练与辅助脚本
└── docs/                  # 文档与样本数据
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆并进入目录
git clone <repository_url>
cd finance-risk-rag

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境

创建 `.env` 文件并填入你的 API 密钥：

```env
OPENAI_API_KEY=sk-...
# 或
MOONSHOT_API_KEY=...
LLM_BASE_URL=https://api.moonshot.cn/v1
```

---

## 🛠️ 命令行工具

系统提供统一的入口 `main.py`：

### 第一步：文档处理 (OCR & 分类)
将 PDF 放入 `docs/` 目录后运行：
```bash
python main.py process
```

### 第二步：风险实体提取
```bash
python main.py extract --input docs/all_extracted.txt
```
*注：此命令支持交互式问答模式。*

### 第三步：RAG 智能问答
构建索引并提问：
```bash
# 构建/更新索引
python main.py query --build

# 提问
python main.py query --ask "该公司的流动性风险状况如何？"
```

---

## 🧪 测试

运行完整测试套件：
```bash
export PYTHONPATH=.
pytest tests/
```

---

## 📄 许可证

本项目采用 MIT 许可证。
