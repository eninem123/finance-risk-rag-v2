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

Finance-Risk-RAG 是一套**企业级财务文本风控 AI 系统**，专为金融机构设计。它整合了先进的 OCR 技术、基于 BERT 的深度学习实体提取、专家规则引擎以及 RAG（检索增强生成）技术，提供从原始 PDF 到深度风险洞察的全流程自动化解决方案。

---

## 系统架构

本系统采用分层架构设计，确保了高可用性与可扩展性：

1.  **数据接入层 (OCR & Processing)**: 使用 Tesseract 与 pdfplumber 结合，支持扫描件与电子档 PDF 的高精度文本提取，内置图像预处理算法。
2.  **认知分析层 (Classification & Extraction)**:
    *   **智能分类**: 利用大语言模型 (LLM) 对文档进行自动归类。
    *   **混合提取**: 结合 BERT 模型（深度语义）与专家规则引擎（高精确度），通过智能仲裁机制识别财务风险实体。
3.  **知识发现层 (RAG Engine)**: 基于 ChromaDB 向量数据库，实现海量财务文档的语义检索与专业问答。
4.  **展示与应用层 (UI & API)**: 包含统一的 CLI 工具、自动化报告生成器以及交互式 Streamlit 仪表盘。

---

## v2.3 核心能力

| 模块 | 说明 |
|------|------|
| `RiskAnalysisService` | **延迟加载**业务编排层，提供标准化的 `run_full_analysis` 接口。 |
| 混合仲裁引擎 | **Score-based Arbitration**: 自动平衡 AI 语义识别与规则匹配。 |
| 鲁棒性 OCR | 增强的异常捕获机制，支持单页失败自动跳过与错误记录。 |
| `dashboard.py` | 交互式面板，支持实时风险监控与 RAG 深度溯源。 |
| 自动化报告 | 一键生成包含风险等级、实体明细及专家建议的 Markdown 报告。 |

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
│   ├── service.py          # 业务编排服务层 (v2.2)
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
