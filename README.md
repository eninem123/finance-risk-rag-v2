# Finance-Risk-RAG v2.1

<div align="center">

**银行级多语言财务文本风控AI系统**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

</div>

---

## 🚀 核心优势

Finance-Risk-RAG 是一套专为金融机构设计的自动化风控系统。它集成了 OCR 识别、NLP 实体抽取与 RAG（检索增强生成）技术，能够高效处理海量财务报表、审计报告及行业研究。

### 核心能力
- **自动化文档处理**: 支持 PDF 到结构化文本的转换，内置 OCR 图像增强引擎。
- **混合动力提取**: 结合行业专家规则与 BERT 深度学习模型，精准识别 17 类金融风险实体。
- **智能风险问答**: 基于语义向量库的 RAG 引擎，支持对复杂风控问题的即时解答。
- **企业级架构**: 模块化设计，高可维护性，支持增量处理与完善的日志审计。

---

## 🛠️ 快速开始

### 安装环境

```bash
# 克隆项目
git clone https://github.com/eninem123/finance-risk-rag-v2.git
cd finance-risk-rag-v2

# 安装依赖
pip install -e .
```

### 统一命令行入口

项目提供统一的 CLI 入口 `main.py`：

```bash
# 1. 批量处理 PDF 文档 (OCR + 分类)
python main.py process

# 2. 识别风险实体
python main.py extract --input docs/all_extracted.txt

# 3. 风险问答
python main.py query "这家公司的流动性风险如何？" --build
```

---

## 🏗️ 架构设计

```
finance-risk-rag/
├── src/
│   └── finance_risk_rag/
│       ├── config.py           # 集中配置管理
│       ├── exceptions.py       # 自定义异常体系
│       ├── models.py           # 统一数据模型
│       ├── utils.py            # 核心工具库
│       ├── extract_text.py     # OCR 与文档处理
│       ├── extract_entities.py # 实体提取管道
│       └── rag_core.py         # RAG 检索引擎
├── tests/                      # 单元测试
├── docs/                       # 系统文档与图像
├── knowledge_base/             # 行业规则库
└── main.py                     # 统一入口
```

---

## 🧪 质量保障

我们使用 `pytest` 确保核心逻辑的稳定性：

```bash
# 运行所有测试
pytest tests/
```

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。
