<div align="center">

# 🏦 Finance-Risk-RAG v2.5

**银行级多语言财务文本风控 AI 系统**
**Professional Enterprise-Grade Financial Risk AI Analytics System**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code_Style-Black-000000?style=for-the-badge)](https://github.com/psf/black)
[![RAG](https://img.shields.io/badge/RAG-Retriever-FF6B6B?style=for-the-badge)]()
[![BERT](https://img.shields.io/badge/NLP-BERT-yellow?style=for-the-badge)]()

**智能 OCR 识别 · BERT 深度提取 · 自动风险评估 · RAG 辅助决策**

</div>

---

## 🎯 项目愿景

Finance-Risk-RAG 是一套专为**银行、保险、审计及合规部门**打造的专业级财务风险控制系统。

系统针对金融领域复杂的 PDF 报表、扫描件、审计报告等非结构化数据，通过 **Service Orchestration (服务编排)** 架构，实现了从原始文档到深度风险报告的全流程自动化分析。

> 🛡️ **核心使命**：将资深风控专家的分析逻辑转化为可扩展的 AI 工作流，显著降低贷前/贷后审核的人为疏漏。

---

## ✨ 核心能力

### 🏗️ 工业级服务编排 (New in v2.5)
- **RiskAnalysisService**: 统一业务层，协调 OCR、分类、实体识别与报告生成。
- **Pipeline 模式**: 模块化设计，支持根据业务需求灵活扩展提取规则。

### 📑 深度文档处理
- **Hybrid OCR Engine**: 结合 `pdfplumber` 文本提取与 `Tesseract 5.x` 图像识别。
- **Image Enhancement**: 自动亮度、对比度优化及去噪，显著提升扫描件识别率。
- **AI Classification**: 自动识别审计报告、行业报告、财报等 6+ 类金融文档。

### 🔍 精准风险实体识别
- **Dual-Engine Extraction**: 规则引擎（高准确度）+ BERT 模型（高泛化性）协同工作。
- **Position-Based Arbitration**: 基于字符偏移量的智能去重与仲裁，处理重叠实体。
- **Multi-Class Support**: 精准识别风险点、金额、组织架构、关键人物等 17 类实体。

### 🧠 RAG 决策辅助
- **Semantic Search**: 基于 ChromaDB 与 ONNX 嵌入模型的极速向量检索。
- **Traceable Q&A**: 所有问答均附带原始文档来源，确保分析的可追溯性。
- **Expert System Integration**: 内置专家提示词工程，生成的回答具备专业金融水准。

---

## 📊 业务价值

| 维度 | 传统模式 | Finance-Risk-RAG | 提升 |
|:--- |:--- |:--- |:---:|
| **处理耗时** | 4-6 小时/份 | < 2 分钟/份 | **~98%** |
| **覆盖深度** | 抽样检查 | 100% 全量扫描 | **全面性** |
| **一致性** | 易受主观影响 | 标准化量化评分 | **客观性** |
| **合规性** | 手工记录 | 自动生成审计轨迹 | **规范化** |

---

## 🚀 快速上手

### 1. 环境准备
```bash
# 克隆仓库
git clone https://github.com/eninem123/finance-risk-rag-v2.git
cd finance-risk-rag-v2

# 安装依赖
pip install -r requirements.txt

# 设置 API 密钥 (可选，用于分类与 Q&A)
export MOONSHOT_API_KEY="your_api_key_here"
```

### 2. 生成风险报告 (核心功能)
```bash
# 为单个 PDF 生成 Markdown 报告
python main.py report --input docs/audit_report_2023.pdf --output-dir ./reports

# 批量分析整个目录
python main.py report --input ./docs/ --output-dir ./reports
```

### 3. RAG 智能问答
```bash
# 构建向量索引
python main.py query "该公司的流动比率是否存在异常？" --build

# 连续查询 (无需重复 --build)
python main.py query "报告中提到的担保风险有哪些？"
```

---

## 📂 模块化架构

```
src/finance_risk_rag/
├── service.py      # 🌟 核心：风险分析编排服务
├── processor.py    # 文档处理 (OCR, Image, Classification)
├── extractor.py    # 实体识别 (Rule, BERT, Arbitration)
├── engine.py       # RAG 检索与问答核心
├── llm.py          # LLM 客户端封装 (带重试机制)
├── config.py       # 系统配置与环境变量
├── models.py       # 统一数据契约
├── utils.py        # 文本处理与通用工具
└── exceptions.py   # 业务级异常体系
```

---

## 🛠️ 技术栈

| 领域 | 选型 |
|:--- |:--- |
| **文本分析** | Python 3.12, NLTK, Jieba, Regex |
| **深度学习** | Transformers (BERT), PyTorch |
| **向量存储** | ChromaDB, ONNX |
| **大模型** | OpenAI API / Moonshot AI |
| **OCR/图像** | Tesseract OCR, pdfplumber, PIL |
| **工程化** | Pytest, Black, Mypy, Flake8 |

---

## 🧪 质量保证

- **单元测试**: `pytest tests/` 覆盖核心逻辑。
- **静态检查**: `mypy`, `flake8` 确保代码类型安全与风格统一。
- **日志体系**: 详尽的 `logs/` 记录，便于回溯处理失败的原因。

---

## 📄 许可证

本项目遵循 [MIT License](LICENSE)。

---

<div align="center">

**为金融风控注入 AI 的力量**

Made with ❤️ for Financial Excellence

</div>
