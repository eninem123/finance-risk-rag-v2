# Finance-Risk-RAG v2.3

<div align="center">

**🏦 银行级多语言财务文本风控 AI 系统**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?style=flat-square)](https://docs.pydantic.dev/)
[![Version](https://img.shields.io/badge/Version-v2.3-blue?style=flat-square)]()

**战略级财务风险管控 · 高度可审计 · 企业级架构**

[核心能力](#核心能力) | [技术架构](#技术架构) | [快速开始](#快速开始) | [企业部署](#企业部署)

</div>

---

## 💎 战略价值

在现代金融监管环境下，对海量非结构化财务文档（审计报告、财报、行业分析）进行高效、准确的风险识别是银行及金融机构的核心竞争力。**Finance-Risk-RAG** 旨在解决以下痛点：

- **自动化合规审核**：大幅降低人工审阅成本，提升合规覆盖率。
- **深层风险穿透**：通过 BERT 与规则引擎双重验证，识别隐藏在文字中的关联风险。
- **辅助决策支持**：基于 RAG 的智能问答系统，为风控人员提供即时的决策参考。

## 🚀 v2.3 核心升级

- **Pydantic 架构迁移**：全系统配置与模型采用 Pydantic v2，实现严格的类型校验与秒级序列化。
- **健壮性增强**：引入指数退避重试机制与领域级异常捕获处理。
- **专业报告引擎**：生成具备“执行摘要”与“专家建议”的标准化风险分析报告。
- **环境隔离**：支持完整的环境变量配置（`.env`），适配容器化部署需求。

## 🛠 技术架构

系统采用模块化分层设计，确保各组件可独立扩展：

1.  **数据接入层 (Processor)**：集成 `pdfplumber` 与 `pytesseract` OCR，支持复杂格式 PDF。
2.  **风险引擎层 (Extractor)**：
    - **Rule-based**: 针对特定金融敏感词的精准匹配。
    - **BERT-based**: 利用预训练语言模型进行语义级实体识别。
3.  **知识检索层 (RAG)**：基于 `ChromaDB` 向量数据库与 `ONNX` 嵌入模型，实现高效上下文检索。
4.  **业务编排层 (Service)**：`RiskAnalysisService` 统一调度，实现端到端闭环。

## 📦 快速开始

### 1. 环境准备
```bash
git clone https://github.com/eninem123/finance-risk-rag-v2.git
cd finance-risk-rag-v2
pip install -r requirements.txt
```

### 2. 配置
拷贝 `.env.example` 并配置您的 API Key：
```bash
cp .env.example .env
# 编辑 .env 文件，填写 MOONSHOT_API_KEY 或 OPENAI_API_KEY
```

### 3. 运行分析
```bash
# 生成专业分析报告
python main.py report --input ./docs/sample_audit.pdf

# 启动交互式 Dashboard
python main.py dashboard
```

## 🏗 企业部署建议

- **OCR 优化**：在生产环境中，建议将 `Tesseract` 替换为更高性能的云端 OCR API 或 PaddleOCR。
- **向量数据库**：对于千万级数据，建议连接外部 `Milvus` 或 `Pinecone` 实例。
- **安全性**：确保 API 密钥存储在受保护的环境变量中，避免硬编码。

---

## 📜 许可证
本项目采用 [MIT License](LICENSE)。

> **免责声明**：本系统生成的分析结果仅供参考，不作为最终投资或信贷决策的唯一依据。
