# Finance-Risk-RAG v2.1

<div align="center">

**银行级多语言财务文本风控AI系统**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

</div>

---

## 📋 目录

- [项目简介](#项目简介)
- [核心功能](#-核心功能)
- [技术架构](#-技术架构)
- [快速开始](#-快速开始)
- [使用指南](#-使用指南)
- [开发与测试](#-开发与测试)
- [许可证](#-许可证)

---

## 项目简介

Finance-Risk-RAG 是一套专为金融领域设计的智能风控系统。它能够批量处理 PDF 财务文档，利用 OCR 技术提取文本，通过 BERT 与规则引擎识别风险实体，并结合 RAG（检索增强生成）技术提供智能风险问答能力。

### 适用场景

- **贷前审查**：自动分析借款企业财务报表中的潜在风险。
- **贷后监控**：持续跟踪企业公告与财务变动。
- **行业分析**：从海量研报中快速检索关键风险信息。

---

## ✨ 核心功能

- 📑 **智能 OCR 提取**：支持高分辨率图像增强与 Tesseract OCR，精准识别扫描件。
- 🔍 **多维实体识别**：结合 BERT 模型与预定义规则，识别 17 类金融风险实体。
- 🧠 **RAG 风险问答**：基于 ChromaDB 向量库，提供精准、可溯源的风险咨询。
- 🏗️ **模块化架构**：面向对象的代码设计，易于集成与扩展。
- 🚀 **统一命令行**：简洁的 CLI 接口，一键完成“处理-提取-查询”全流程。

---

## 技术架构

```
finance-risk-rag/
├── src/finance_risk_rag/   # 核心源码包
│   ├── config.py           # 集中配置管理
│   ├── models.py           # 统一数据模型
│   ├── exceptions.py       # 自定义异常体系
│   ├── extract_text.py     # OCR 与文档分类
│   ├── extract_entities.py # 实体提取流水线
│   ├── rag_core.py         # RAG 引擎核心
│   └── utils.py            # 工具函数库
├── research/               # 实验与研究脚本
├── tests/                  # 单元测试 suite
├── docs/                   # 输入文档与结果输出
├── main.py                 # 统一命令行入口
└── requirements.txt        # 依赖清单
```

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- Tesseract OCR (需安装并配置路径)

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/eninem123/finance-risk-rag-v2.git
cd finance-risk-rag-v2

# 安装依赖
pip install -r requirements.txt
```

### 环境变量配置

创建 `.env` 文件：

```env
OPENAI_API_KEY=your_key_here
LLM_PROVIDER=moonshot
LLM_BASE_URL=https://api.moonshot.cn/v1
```

---

## 📖 使用指南

项目提供统一的 `main.py` 入口：

```bash
# 1. 文本提取与 OCR 分类
python main.py process

# 2. 风险实体提取
python main.py extract --input docs/all_extracted.txt --output docs/risk_report.json

# 3. 构建索引并提问
python main.py query --build "这家公司的流动性风险如何？"
```

---

## 🧪 开发与测试

我们建议在开发过程中运行单元测试：

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
pytest tests/
```

---

## 许可证

本项目采用 MIT 许可证。
