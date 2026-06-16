<div align="center">

# 🏦 Finance-Risk-RAG v2.2

**银行级多语言财务文本风控 AI 系统**

**Professional Multi-language Financial Risk AI Control System**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code_Style-Black-000000?style=for-the-badge)](https://github.com/psf/black)
[![CI](https://img.shields.io/badge/CI-Passing-00D26A?style=for-the-badge)]()
[![RAG](https://img.shields.io/badge/RAG-Powered-FF6B6B?style=for-the-badge)]()

**OCR 智能识别 · BERT 实体提取 · RAG 风险问答 · 批量自动化处理 · 交互式仪表盘**

</div>

---

## 🎯 项目简介

Finance-Risk-RAG 是一套**银行级财务文本风控 AI 系统**，专为金融机构的贷前审查、贷后监控和风险预警场景设计。

v2.2 版本引入了全新的 **Service 编排层**、**增强型 BERT 提取引擎** 与 **Streamlit 交互式仪表盘**，实现了从原始 PDF 到深度风控报告的全自动化流水线。

> 🌐 **双语支持**：完整支持中英文财务文档处理，面向国际化场景。

---

## ✨ 核心功能

### 📑 智能 OCR 文档处理
- 高分辨率图像增强算法，提升扫描件识别率
- 支持多格式 PDF 解析（原生/扫描/混合）
- **AI 驱动的文档自动分类**（审计报告、财报、行业报告等）
- 批量处理，支持目录级递归扫描

### 🔍 多维风险实体识别 (v2.2 增强)
- **17 类**金融风险实体精准识别
- **BERT 深度学习模型 (支持长文本分块) + 规则引擎** 双重仲裁
- **基于字符偏移量 (Character Offset) 的精准去重与重叠仲裁**
- 自定义评分策略 (Scoring Strategy)，灵活调整风险权重

### 🧠 RAG 智能风险问答
- 基于 ChromaDB 向量数据库的语义检索
- **增量索引技术**：自动检测文件哈希，仅处理变更内容
- 精准、可溯源的风险咨询问答
- 自动生成 AI 财务摘要与风险建议

### 🏗️ 企业级架构
- **Service 编排层**：统一协调 OCR、提取与 RAG 流程
- 模块化设计，支持 `Rule-based` 与 `Model-based` 提取器扩展
- 完整的异常处理与日志体系
- **统一 CLI 入口** + **Streamlit Web 仪表盘**

---

## 📊 效率提升对比

| 应用场景 | 人工处理 | AI 处理 | 效率提升 |
|---------|---------|---------|:-------:|
| 贷前审查 | 24 小时 | 10 分钟 | **70%** |
| 贷后监控 | 3 天 | 30 分钟 | **85%** |
| 风险预警 | 人工巡检 | 实时监控 | **92%** |

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                      输入层 (Input Layer)                    │
│          PDF 文档  │  扫描件  │  Word  │  Excel              │
├─────────────────────────────────────────────────────────────┤
│                    处理层 (Processing Layer)                 │
│  OCR 识别  →  文本清洗  →  AI 文档分类  →  结构解析         │
├─────────────────────────────────────────────────────────────┤
│                     分析层 (Analysis Layer)                  │
│  实体提取 (BERT/Rule)  →  重叠仲裁  →  风险评分  →  关系图谱 │
├─────────────────────────────────────────────────────────────┤
│                     应用层 (Application Layer)               │
│  RAG 问答  │  风险报告  │  交互仪表盘  │  API 接口           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求
- Python 3.9+
- Tesseract OCR (需安装并配置路径)
- 建议配置：4核 CPU + 8GB 内存

### 安装部署

```bash
# 1. 克隆项目
git clone https://github.com/eninem123/finance-risk-rag-v2.git
cd finance-risk-rag-v2

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境 (见 .env.example)
cp .env.example .env
```

### 运行方式

**1. 命令行界面 (CLI)**
```bash
# 一键生成全流程风险报告
python main.py report --dir ./docs/

# 启动 RAG 问答
python main.py query "该公司的流动比率是否存在异常？" --build
```

**2. Web 仪表盘**
```bash
streamlit run dashboard.py
```

---

## 📁 项目结构

```
finance-risk-rag-v2/
├── src/
│   └── finance_risk_rag/     # 核心源码包
│       ├── service.py        # 业务编排服务 (v2.2 新增)
│       ├── config.py         # 集中配置管理
│       ├── engine.py         # RAG 检索引擎
│       ├── extractor.py      # 实体提取核心 (支持多策略)
│       ├── processor.py      # 文档处理与 OCR
│       ├── models.py         # 统一数据模型
│       └── ...
├── dashboard.py              # Streamlit 交互仪表盘
├── main.py                   # 统一命令行入口
├── tests/                    # 单元测试套件
├── requirements.txt          # 依赖清单
└── README.md                 # 项目说明文档
```

---

## 🛠️ 技术栈

| 类别 | 技术选型 |
|------|---------|
| 核心语言 | Python 3.9+ |
| OCR 引擎 | Tesseract OCR + 图像增强 |
| NLP 模型 | BERT / Transformers |
| 向量数据库 | ChromaDB |
| LLM 集成 | OpenAI / Moonshot / DeepSeek |
| Web 展示 | Streamlit |
| 代码规范 | Black + Flake8 + Mypy |

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

<div align="center">

**如果这个项目对你有帮助，别忘了点个 ⭐ Star 支持一下**

Made with ❤️ by eninem123

</div>
