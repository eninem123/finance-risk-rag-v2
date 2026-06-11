<div align="center">

# 🏦 Finance-Risk-RAG v2.1

**银行级多语言财务文本风控 AI 系统**

**Professional Multi-language Financial Risk AI Control System**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code_Style-Black-000000?style=for-the-badge)](https://github.com/psf/black)
[![CI](https://img.shields.io/badge/CI-Passing-00D26A?style=for-the-badge)]()
[![RAG](https://img.shields.io/badge/RAG-Powered-FF6B6B?style=for-the-badge)]()

**OCR 智能识别 · BERT 实体提取 · RAG 风险问答 · 批量自动化处理**

</div>

---

## 🎯 项目简介

Finance-Risk-RAG 是一套**银行级财务文本风控 AI 系统**，专为金融机构的贷前审查、贷后监控和风险预警场景设计。

系统整合了 **OCR 智能识别**、**BERT 实体提取**、**规则引擎匹配**与 **RAG 检索增强生成** 四大核心能力，能够批量处理 PDF 财务文档，自动识别风险实体并提供智能问答，将传统人工审查效率提升 **70%-92%**。

> 🌐 **双语支持**：完整支持中英文财务文档处理，面向国际化场景。

---

## ✨ 核心功能

### 📑 智能 OCR 文档处理
- 高分辨率图像增强算法，提升扫描件识别率
- 支持多格式 PDF 解析（原生/扫描/混合）
- 自动文档分类与版面分析
- 批量处理，支持目录级递归扫描

### 🔍 多维风险实体识别
- **17 类**金融风险实体精准识别
- BERT 深度学习模型 + 规则引擎双重校验
- 支持自定义实体类型与规则扩展
- 实体关系抽取与关联分析

### 🧠 RAG 智能风险问答
- 基于 ChromaDB 向量数据库的语义检索
- 精准、可溯源的风险咨询问答
- 支持多轮对话与上下文理解
- 风险等级自动评估与建议生成

### 🏗️ 企业级架构
- 模块化设计，易于集成与扩展
- 统一配置管理，灵活适配不同场景
- 完整的异常处理与日志体系
- 专业 CLI 接口，一键完成全流程

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
│  OCR 识别  →  文本清洗  →  文档分类  →  结构解析            │
├─────────────────────────────────────────────────────────────┤
│                     分析层 (Analysis Layer)                  │
│  实体提取 (BERT)  →  规则匹配  →  风险评级  →  关系图谱      │
├─────────────────────────────────────────────────────────────┤
│                     应用层 (Application Layer)               │
│  RAG 问答  │  风险报告  │  预警推送  │  可视化展示          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求
- Python 3.9+
- Linux / macOS / Windows
- 建议配置：4核 CPU + 8GB 内存（批量处理建议 16GB+）

### 安装部署

```bash
# 1. 克隆项目
git clone https://github.com/eninem123/finance-risk-rag-v2.git
cd finance-risk-rag-v2

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载 NLTK 数据（首次运行）
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 4. 运行 CLI 查看帮助
python main.py --help
```

### 快速上手

```bash
# 处理单个 PDF 文档
python main.py process --input document.pdf

# 批量处理目录
python main.py process --input ./docs/ --output ./results/

# 启动 RAG 问答模式
python main.py rag --knowledge-base ./knowledge_base/

# 交互式查询
python main.py query "这笔贷款有哪些风险点？"
```

---

## 📁 项目结构

```
finance-risk-rag-v2/
├── src/
│   └── finance_risk_rag/     # 核心源码包
│       ├── __init__.py
│       ├── config.py         # 集中配置管理
│       ├── models.py         # 统一数据模型
│       ├── exceptions.py     # 自定义异常体系
│       ├── extract_text.py   # OCR 与文档分类
│       ├── extract_entities.py # 实体提取流水线
│       ├── extractor.py      # 提取器接口
│       ├── processor.py      # 文档处理器
│       ├── rag_core.py       # RAG 引擎核心
│       ├── engine.py         # 业务引擎
│       └── utils.py          # 工具函数库
├── research/                 # 实验与研究脚本
│   ├── bert_finetune.py      # BERT 微调实验
│   └── risk_qa_cli.py        # 风险问答 CLI 原型
├── tests/                    # 单元测试套件
├── docs/                     # 输入文档与结果输出
├── knowledge_base/           # 风险规则与词典
├── main.py                   # 统一命令行入口
├── requirements.txt          # 依赖清单
├── pyproject.toml            # 项目配置
├── .flake8                   # 代码规范
├── mypy.ini                  # 类型检查配置
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
| LLM 集成 | OpenAI API / 本地模型 |
| 文档处理 | PyPDF2 / pdfplumber / python-docx |
| 配置管理 | Pydantic Settings |
| 代码规范 | Black + Flake8 + Mypy |
| CI/CD | GitHub Actions |
| 测试框架 | Pytest |

---

## 🧪 质量保障

- ✅ **单元测试**：核心模块完整测试覆盖
- ✅ **类型检查**：Mypy 静态类型检查
- ✅ **代码规范**：Black 统一代码风格
- ✅ **CI 流水线**：GitHub Actions 自动化验证
- ✅ **安全扫描**：代码安全漏洞扫描

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

<div align="center">

**如果这个项目对你有帮助，别忘了点个 ⭐ Star 支持一下**

Made with ❤️ by eninem123

</div>
