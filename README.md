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

> 本仓库已升级至 v2.3，引入了更强大的提取引擎、LLM 执行摘要生成以及交互式可视化分析功能。

---

## v2.3 核心能力

| 模块 | 说明 |
|------|------|
| `RiskAnalysisService` | 业务编排层，新增标准化 API 响应与 LLM 执行摘要生成 |
| `dashboard.py` | Streamlit 仪表盘，支持风险分布图表、报告预览与下载 |
| `BERT 滑动窗口` | 支持长文本分块处理 (Overlap Chunking)，解决长文档识别限制 |
| `冲突仲裁引擎` | 基于置信度与权重的混合提取仲裁逻辑，显著提升识别精度 |
| `自动化报告` | 一键生成包含执行摘要、风险评估、详细实体及专家建议的综合报告 |

---

## 快速开始

```bash
git clone https://github.com/eninem123/finance-risk-rag-v2.git
cd finance-risk-rag-v2
pip install -r requirements.txt

# 全流程处理
python main.py process --input ./docs/

# RAG 问答
python main.py query "这笔贷款有哪些风险点？"

# 生成风险报告
python main.py report --input document.pdf --output-dir reports/

# 启动可视化面板
python main.py dashboard
```

---

## 项目结构

```
finance-risk-rag-v2/
├── src/finance_risk_rag/
│   ├── service.py          # 业务编排服务层 (v2.3)
│   ├── extractor.py        # 实体提取管道
│   ├── processor.py        # 文档 OCR 处理
│   ├── engine.py           # RAG 引擎
│   └── ...
├── dashboard.py            # Streamlit 可视化面板
├── main.py                 # 统一 CLI 入口
├── tests/                  # 单元测试
└── .github/workflows/      # CI（仅 PR → main 触发）
```

---

## 2026-06 仓库整理记录

### 合并 PR（#11–#17 → main）

| PR | 分支 | 内容 |
|----|------|------|
| #11 | `feature/professional-v2.1-optimization-*` | v2.1 架构优化 |
| #12 | `feature/enterprise-optimization-*` | 企业服务层 + 报告生成 |
| #13 | `optimize-finance-risk-rag-v2.2-*` | v2.2 服务层重构 |
| #14 | `feature/professional-refactoring-and-dashboard-*` | 服务层 + 仪表盘 |
| #15 | `feature/v2.2-optimization-7723*` | v2.2 架构升级 |
| #16 | `feature/optimize-architecture-v2.2-*` | 架构优化 + Dashboard |
| #17 | `feature/v2.2-optimization-1572*` | v2.2 全面升级 |
| #18 | `feature/professional-refactoring-and-dashboard-v2.2-*` | 已合并至 main（2026-06-20） |
| #19 | `feature/professional-refactoring-and-dashboard-v2.2-*` | **当前唯一迭代入口（Draft PR）** |

### 删除分支（8 个 feature 临时分支）

- `feature/enterprise-optimization-5050392069136229718`
- `feature/optimize-architecture-v2.2-18168074359074526671`
- `feature/professional-refactoring-and-dashboard-4562812990206762378`
- `feature/professional-refactoring-and-dashboard-v2.2-2030757817022327085`
- `feature/professional-v2.1-optimization-5571732041245732993`
- `feature/v2.2-optimization-15723274487901237107`
- `feature/v2.2-optimization-7723375731582770803`
- `optimize-finance-risk-rag-v2.2-647354061673373050`

### CI 降噪变更

- 移除 `push` 自动触发，**仅 `pull_request → main`** 时运行
- 取消 Python 多版本矩阵与 lint / test 重复步骤
- 简化为单版本基础编译校验（`py_compile` + `compileall`）

### Bot 降噪变更

- 新增 `.cursor/rules/bot-noise-reduction.mdc`：禁止自动 PR 评论与 CI 告警回复，仅响应手动 @

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE)
