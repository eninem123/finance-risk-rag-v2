# Finance-Risk-RAG v2.3

<div align="center">

**银行级多语言财务文本风控 AI 系统**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/badge/Version-v2.3-blue?style=flat-square)]()

OCR 智能识别 · BERT 实体提取 · RAG 风险问答 · Pydantic 架构 · AI 执行摘要

</div>

---

## 项目简介

Finance-Risk-RAG 是一套**银行级财务文本风控 AI 系统**，整合 OCR、BERT 实体提取、规则引擎与 RAG 检索增强生成，支持批量 PDF 处理、风险实体识别与智能问答。

**v2.3 版本引入了基于 Pydantic 的专业架构、AI 生成的执行摘要以及增强的风险分析报告。**

---

## v2.3 核心升级

| 模块 | 说明 |
|------|------|
| **Pydantic 架构** | 全面改用 `Pydantic` 进行模型校验与 `pydantic-settings` 配置管理。 |
| **AI 执行摘要** | 自动为每份财务文档生成 150 字以内的风控执行摘要（由 LLM 驱动）。 |
| **增强型报告** | Markdown 报告新增 AI 摘要、结构化实体分布表与分级风控建议。 |
| **BERT 滑动窗口** | 支持长文本切片提取，解决 BERT 512 字符限制问题。 |
| **混合提取仲裁** | 改进的规则引擎与 BERT 实体碰撞仲裁机制，优先考虑高置信度模型输出。 |
| **可视化 Dashboard** | 新增“风险报告”页面，支持全流程分析预览与 Markdown 下载。 |

---

## 快速开始

```bash
git clone https://github.com/eninem123/finance-risk-rag-v2.git
cd finance-risk-rag-v2
pip install -r requirements.txt
pip install pydantic pydantic-settings

# 全流程处理并构建索引
python main.py process --input ./docs/

# 生成 AI 风险报告
python main.py report --input ./docs/sample.pdf --output-dir reports/

# 执行 RAG 智能问答
python main.py query "该公司的财务风险主要集中在哪些方面？"

# 启动 Streamlit 可视化面板
python main.py dashboard
```

---

## 项目结构

```
finance-risk-rag-v2/
├── src/finance_risk_rag/
│   ├── config.py          # 基于 Pydantic Settings 的配置管理
│   ├── service.py         # 业务编排服务层 (包含 AI 摘要生成)
│   ├── extractor.py       # 混合实体提取管道 (BERT 滑动窗口 + 规则)
│   ├── models.py          # Pydantic 数据模型定义
│   ├── engine.py          # RAG 检索增强生成引擎
│   └── ...
├── dashboard.py           # Streamlit 可视化面板 (数据/分析/报告/RAG)
├── main.py                # 统一 CLI 入口 v2.3
├── tests/                 # 单元测试与端到端集成测试
└── .github/workflows/     # CI 流程
```

---

## 开发与质量保证

项目遵循高标准的开发规范：
- **代码格式化**: 使用 `black` (100 字符) 与 `isort`。
- **静态检查**: `flake8` 确保无代码异味。
- **自动化测试**: 包含 `pytest` 单元测试与 `IntegrationWorkflow` 全流程测试。

```bash
# 运行测试
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
pytest tests/
```

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE)
