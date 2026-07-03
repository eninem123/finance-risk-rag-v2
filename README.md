# Finance-Risk-RAG v2.3

<div align="center">

**银行级多语言财务文本风控 AI 系统**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/badge/Version-v2.3-blue?style=flat-square)]()

Pydantic 数据校验 · 智能 PII 脱敏 · 风险矩阵可视化 · 银行级架构优化

</div>

---

## 项目简介

Finance-Risk-RAG 是一套**银行级财务文本风控 AI 系统**。v2.3 版本引入了基于 Pydantic 的严谨数据模型、PII 隐私脱敏机制、自适应 OCR 预处理，以及更加直观的风险矩阵可视化面板。系统整合了 OCR、BERT 实体提取、规则引擎与 RAG 检索增强生成，助力金融机构实现自动化财务风险识别。

---

## v2.3 核心能力与架构升级

| 模块 | 说明 | 升级点 (v2.3) |
|------|------|------|
| **Pydantic Models** | 核心数据模型 | 引入 `BaseModel` 强制类型校验与序列化 |
| **PII Masking** | 隐私安全保护 | 自动脱敏银行卡、身份证等敏感信息 |
| **Scoring Strategy** | 风险评估策略 | 抽象 `ScoringStrategy` 模式，支持自定义量化逻辑 |
| **Risk Matrix** | 可视化分析 | Dashboard 新增 Plotly 风险矩阵 (置信度 vs 评分) |
| **Adaptive OCR** | 图像预处理 | 优化图像增强算法，提升扫描件识别准确率 |
| **Lazy Service** | 业务编排层 | 懒加载模式优化资源占用，加速系统启动 |

---

## 快速开始

```bash
git clone https://github.com/eninem123/finance-risk-rag-v2.git
cd finance-risk-rag-v2

# 安装依赖
pip install -r requirements.txt

# 启动可视化面板
python main.py dashboard

# 命令行全流程分析
python main.py report --input ./docs/document.pdf --output-dir reports/
```

---

## 项目结构

```
finance_risk_rag/
├── src/finance_risk_rag/
│   ├── service.py          # 业务编排服务层 (懒加载模式)
│   ├── extractor.py        # 实体提取管道 (策略模式)
│   ├── processor.py        # 文档 OCR & 智能分类
│   ├── models.py           # Pydantic 数据模型
│   ├── config.py           # Pydantic Settings 配置
│   ├── llm.py              # LLM 客户端 (含 PII 脱敏)
│   └── utils.py            # 工具类 (正则、哈希、PII 掩码)
├── dashboard.py            # Streamlit & Plotly 可视化
├── main.py                 # 统一 CLI 入口
└── tests/                  # 完善的单元测试
```

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE)
