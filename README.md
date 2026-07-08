# Finance-Risk-RAG v2.3

<div align="center">

**银行级多语言财务文本风控 AI 系统**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/badge/Version-v2.3-blue?style=flat-square)]()

OCR 智能识别 · BERT 实体提取 · RAG 风险问答 · Plotly 风险矩阵 · AI 执行摘要

</div>

---

## 项目简介

Finance-Risk-RAG 是一套**银行级财务文本风控 AI 系统**，整合 OCR、BERT 实体提取、规则引擎与 RAG 检索增强生成。v2.3 版本引入了更专业的架构设计、AI 驱动的风险摘要以及交互式风险矩阵可视化。

---

## v2.3 核心改进

- **专业架构重构**: 引入 `BaseExtractor` 与 `ScoringStrategy` 设计模式，支持提取逻辑与风险评分逻辑的解耦。
- **AI 执行摘要**: 在生成的风险报告中，利用 LLM 自动生成针对银行信贷/投资视角的执行摘要。
- **交互式风险矩阵**: Dashboard 新增 Plotly 驱动的“风险矩阵”页面，可视化展示风险实体的“影响度 vs 置信度”。
- **PII 脱敏保护**: 内置 `PIIMasker` 工具，在调用外部 LLM 前自动对敏感信息（如手机号、银行卡、身份证）进行掩码处理。
- **报告模板升级**: 生成的 Markdown 报告升级为“银行级”专业排版，包含流水号、专家结论与针对性风控建议。

---

## 快速开始

```bash
git clone https://github.com/eninem123/finance-risk-rag-v2.git
cd finance-risk-rag-v2
pip install -r requirements.txt

# 1. 全流程处理 (OCR + 分类 + 索引)
python main.py process --input ./docs/

# 2. 生成银行级风险报告 (含 AI 摘要)
python main.py report --input document.pdf --output-dir reports/

# 3. 启动 v2.3 可视化面板 (包含风险矩阵)
python main.py dashboard
```

---

## 系统架构

```
src/finance_risk_rag/
├── service.py          # 业务编排服务层 (v2.3 增强)
├── extractor.py        # 实体提取管道 (ScoringStrategy 模式)
├── processor.py        # 文档 OCR 处理 (LLM 文档分类)
├── engine.py           # RAG 引擎 (ChromaDB + ONNX)
├── llm.py              # LLM 客户端封装 (PII 保护)
├── models.py           # 数据模型 (Entity, ExtractionResult)
└── utils.py            # 工具库 (PII Masking, Text Cleaning)
```

---

## 核心配置

系统支持通过 `.env` 或 `src/finance_risk_rag/config.py` 进行配置：
- `risk_level_low/medium/high`: 风险等级阈值设定。
- `llm_provider`: 支持 OpenAI, Moonshot 等主流模型。
- `ocr_languages`: 默认支持中繁、中简、英文。

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE)
