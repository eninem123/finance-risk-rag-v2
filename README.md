# Finance-Risk-RAG v2.1

<div align="center">

**银行级多语言财务文本风控AI系统**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

</div>

---

## 🏗️ 架构升级

Finance-Risk-RAG v2.1 进行了重大的架构升级，从单文件脚本转向了模块化的 Python 包结构，极大提升了系统的可维护性、测试覆盖率和扩展性。

### 核心改进
- **模块化设计**: 核心逻辑拆分为 `processor`, `extractor`, `engine`, `llm` 等独立模块。
- **统一 CLI**: 提供单一入口 `main.py` 统筹所有功能。
- **类型安全**: 全面应用 Python 类型注解。
- **异常处理**: 建立了完善的自定义异常体系。
- **单元测试**: 核心功能均有自动化测试保障。

---

## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone https://github.com/eninem123/finance-risk-rag-v2.git
cd finance-risk-rag-v2

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置
编辑 `.env` 文件（可参考 `.env.example`）：
```env
OPENAI_API_KEY=your_key
LLM_PROVIDER=moonshot
LLM_BASE_URL=https://api.moonshot.cn/v1
```

### 3. 使用命令行 (CLI)

系统提供 `main.py` 作为统一入口：

- **提取文本与分类**:
  ```bash
  python main.py process
  ```
- **提取风险实体**:
  ```bash
  python main.py extract --input docs/all_extracted.txt
  ```
- **RAG 问答 (构建索引并提问)**:
  ```bash
  python main.py query "这家公司的流动性风险如何？" --build
  ```

---

## 📁 项目结构

```
.
├── src/
│   └── finance_risk_rag/       # 核心包
│       ├── engine.py           # RAG 检索引擎
│       ├── extractor.py        # 风险实体提取 (Rule + BERT)
│       ├── processor.py        # OCR 与文档处理
│       ├── llm.py              # LLM 客户端封装
│       ├── models.py           # 数据模型
│       ├── config.py           # 集中配置管理
│       └── utils.py            # 工具函数
├── tests/                      # 单元测试
├── main.py                     # 统一 CLI 入口
├── requirements.txt            # 依赖清单
└── docs/                       # 文档与数据
```

---

## 🧪 测试

使用 `pytest` 运行测试：
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
pytest tests/
```

---

## 许可证

本项目采用 MIT 许可证。
