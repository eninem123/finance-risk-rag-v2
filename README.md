# Finance-Risk-RAG v2.3

<div align="center">

**银行级多语言财务文本风控 AI 系统**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/badge/Version-v2.3-blue?style=flat-square)]()

OCR 智能识别 · BERT 实体提取 · RAG 风险问答 · Streamlit 可视化面板

</div>

---

## 项目简介

Finance-Risk-RAG 是一套**银行级财务文本风控 AI 系统**，整合 OCR、BERT 实体提取、规则引擎与 RAG 检索增强生成，支持批量 PDF 处理、风险实体识别与智能问答。

**v2.3 升级重点**：全面采用 Pydantic v2 模型驱动，引入 pluggable 评分策略，增强脱敏安全（PII Masking），提供更专业的可视化报告。

---

## v2.3 核心能力

| 模块 | 说明 |
|------|------|
| `RiskAnalysisService` | 业务编排层 (v2.3)，支持延迟初始化与标准化输出 |
| `PIIMasker` | 安全增强，自动识别并脱敏敏感金融 PII 数据 |
| `ScoringStrategy` | 可扩展评分策略，支持自定义风险量化逻辑 |
| `dashboard.py` | 增强可视化（Plotly 风险矩阵与分布图） |
| `Pydantic v2` | 全流程数据校验，提升系统健壮性 |

---

## 快速开始

```bash
git clone https://github.com/eninem123/finance-risk-rag-v2.git
cd finance-risk-rag-v2

# 使用 Makefile 快速安装与启动
make install

# 启动可视化面板
make dashboard
```

### CLI 命令

```bash
# 全流程处理
python main.py process --input ./docs/

# RAG 问答
python main.py query "这笔贷款有哪些风险点？"

# 生成银行级风险报告
python main.py report --input document.pdf --output-dir reports/
```

---

## 项目结构

```
finance-risk-rag-v2/
├── src/finance_risk_rag/
│   ├── service.py          # 业务编排服务层 (v2.3)
│   ├── extractor.py        # 实体提取与评分策略
│   ├── processor.py        # 文档 OCR 处理
│   ├── engine.py           # RAG 引擎
│   ├── models.py           # Pydantic v2 数据模型
│   └── utils.py            # 工具类与 PII 脱敏
├── Makefile                # 自动化开发指令
├── dashboard.py            # Streamlit 专业面板
├── main.py                 # 统一 CLI 入口 (v2.3)
└── tests/                  # 单元测试
```

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE)
