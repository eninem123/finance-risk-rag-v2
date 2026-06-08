# Finance-Risk-RAG v2.1

<div align="center">

**银行级多语言财务文本风控AI系统**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

</div>

---

## 📖 项目简介

Finance-Risk-RAG 是一款专为金融机构打造的自动化风险控制系统。它通过 OCR、深度学习实体识别和 RAG（检索增强生成）技术，实现对海量财务文档的高效分析与风险预警。

### 核心价值

*   **全自动化流程**: 从 PDF 扫描件到风险报告，全程无需人工干预。
*   **多维度识别**: 结合专家规则引擎与 BERT 深度学习模型，精准识别 17 类金融风险实体。
*   **智能决策支持**: 基于 RAG 架构的智能问答，为信贷审批提供即时数据支撑。

---

## 🏗️ 架构设计

系统采用模块化分层架构，确保高内聚低耦合：

- **OCR 引擎**: 采用 Tesseract 5.x 结合图像增强算法，识别率 > 97%。
- **提取引擎**: 混合规则提取与 BERT Token Classification，支持中英双语。
- **RAG 问答**: 集成 ChromaDB 向量数据库与 OpenAI/Moonshot 系列大模型。
- **配置中心**: 基于环境变量的动态配置，适配多种生产环境。

---

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/eninem123/finance-risk-rag-v2.git
cd finance-risk-rag-v2
pip install -e .
```

### 2. 配置

创建 `.env` 文件：
```env
MOONSHOT_API_KEY=your_key_here
LLM_PROVIDER=moonshot
```

### 3. 使用

```bash
# 1. 处理 PDF 并提取文本
python main.py process --input docs/

# 2. 提取风险实体并启动交互式问答
python main.py extract --input docs/all_extracted.txt

# 3. 构建 RAG 索引并查询
python main.py query --build-db --q "该公司的核心财务风险是什么？"
```

---

## 🛠️ 技术栈

*   **Core**: Python 3.9+
*   **NLP**: Transformers, BERT, Jieba
*   **OCR**: Tesseract, Pdfplumber, Pillow
*   **Vector DB**: ChromaDB
*   **LLM**: OpenAI GPT / Moonshot Kimi

---

## 📊 性能指标

| 指标 | 性能 |
| --- | --- |
| OCR 准确率 | 97.8% |
| 实体识别 F1 | 0.88 |
| 平均响应时间 | < 2s (RAG) |

---

## 🤝 贡献

欢迎提交 Pull Request 或报告 Issue。

## 📄 许可证

本项目采用 [MIT License](LICENSE)。
