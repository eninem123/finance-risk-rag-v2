# Finance-Risk-RAG v3.0

<div align="center">

**银行级多语言财务文本风控 AI 系统**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

</div>

---

## 📖 项目简介

Finance-Risk-RAG 是一款专业的金融风控系统，利用 Retrieval-Augmented Generation (RAG) 技术，对海量财务文档进行深度分析。系统支持 PDF 解析、OCR 识别、多维实体提取、自动文档分类以及基于语义的风险问答。

### 核心价值
- **自动化审查**: 将数天的贷前/贷后审查工作缩短至分钟级。
- **高精度识别**: 结合 BERT 模型与规则引擎，精准捕捉财务风险点。
- **智能对话**: 通过 RAG 技术，直接就财务报表中的特定风险点进行专业问答。

---

## 🏗️ 架构设计

系统采用高度模块化的专业架构设计：

```
finance_risk_rag/
├── src/
│   └── finance_risk_rag/
│       ├── config.py       # 集中式配置管理
│       ├── processor.py    # 文档处理与 OCR 管道
│       ├── extractor.py    # 混合动力实体识别引擎
│       ├── engine.py       # RAG 核心引擎 (ChromaDB + ONNX)
│       ├── llm.py          # 统一 LLM 客户端
│       ├── models.py       # 强类型数据模型
│       ├── exceptions.py   # 统一异常体系
│       └── utils.py        # 工业级工具函数
├── tests/                  # 完备的单元测试集
├── main.py                 # 统一命令行入口
└── requirements.txt        # 依赖清单
```

---

## 🚀 快速开始

### 1. 环境准备
```bash
# 安装核心依赖
pip install -r requirements.txt
```

### 2. 配置
在 `.env` 中设置您的 API 密钥：
```env
OPENAI_API_KEY=your_key_here
LLM_BASE_URL=https://api.moonshot.cn/v1
```

### 3. 一键运行
```bash
# 1. 文档预处理 (OCR + 分类)
python main.py process

# 2. 风险实体提取
python main.py extract

# 3. 智能风险问答
python main.py query "该公司的流动性风险状况如何？" --build
```

---

## ✨ 核心特性

- **专业级 OCR**: 针对财务报表优化的图像预处理算法。
- **混合抽取引擎**: 规则引擎保证召回，BERT 模型提升精度。
- **企业级 RAG**: 使用 ChromaDB 向量库，支持 ONNX 模型加速，无需昂贵算力。
- **统一入口**: 通过 `main.py` 即可驱动整个风控流水线。

---

## 🧪 质量保证
系统内置完备的测试用例：
```bash
export PYTHONPATH=$PYTHONPATH:.
pytest tests/
```

---

## 🤝 贡献与反馈
欢迎提交 Issue 或 Pull Request 来共同完善这个金融风控利器！
