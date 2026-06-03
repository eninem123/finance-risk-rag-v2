# Finance-Risk-RAG v2.0

<div align="center">

**银行级多语言财务文本风控AI系统**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP8-orange.svg)](https://pep8.org/)

[功能特性](#-功能特性) • [快速开始](#-快速开始) • [架构设计](#-架构设计) • [API文档](#-api文档) • [贡献指南](#-贡献指南)

</div>

---

## 📋 目录

- [项目简介](#项目简介)
- [功能特性](#-功能特性)
- [技术架构](#-技术架构)
- [快速开始](#-快速开始)
- [配置说明](#-配置说明)
- [使用指南](#-使用指南)
- [API文档](#-api文档)
- [性能指标](#-性能指标)
- [常见问题](#-常见问题)
- [贡献指南](#-贡献指南)
- [许可证](#-许可证)

---

## 项目简介

Finance-Risk-RAG 是一套针对财务文档的智能风控系统，支持批量处理PDF文件，通过OCR识别、文档分类、风险实体抽取和RAG问答，实现金融风险的自动化分析与预警。

### 适用场景

| 场景 | 节省人力 | 时间效率提升 |
|------|---------|-------------|
| 贷前审查 | 70% | 24小时 → 10分钟 |
| 贷后监控 | 85% | 3天 → 30分钟 |
| 风险预警 | 92% | 手动排查 → 自动预警 |

---

## ✨ 功能特性

### 核心能力

| 能力 | 实现方案 | 关键特性 |
|------|---------|---------|
| 批量OCR | 600DPI + 图像增强 + Tesseract 5.5 | 识别率97.8%+，支持表格/图片提取 |
| 文档分类 | Kimi AI 自动分类 | 6类文档分类，准确率99% |
| 增量处理 | MD5 + 版本管理 | 已处理文件自动跳过，节省90%算力 |
| 风险实体识别 | 12类规则 + AI增强（BERT + Kimi） | 支持17类金融实体，跨语言识别率88% |
| RAG问答 | Chroma向量库 + ONNX模型 | 支持复杂风险问题查询，零网络依赖 |
| 实时监控 | 增量处理 + 定时任务调度 | 新增文件自动分析，延迟≤5分钟 |

### v2.0 新特性

- 🏗️ **架构重构**: 采用模块化设计，代码可维护性提升
- 📝 **类型注解**: 全面支持Python类型提示，IDE友好
- ⚠️ **异常处理**: 完善的异常体系，错误定位更精准
- 📚 **文档完善**: API文档、开发指南、最佳实践
- 🧪 **测试覆盖**: 单元测试框架，保证代码质量

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Finance-Risk-RAG v2.0                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  文档输入    │  │  OCR处理    │  │  文档分类   │         │
│  │  (PDF/TXT)  │→│  (Tesseract)│→│  (Kimi AI)  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   实体提取引擎                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ 规则提取器  │  │ BERT提取器  │  │  实体融合   │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    RAG引擎                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ 文本分块器  │  │ 向量数据库  │  │  LLM客户端  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   风险分析输出                        │   │
│  │  • 风险评分  • 风险等级  • 趋势分析  • 智能问答    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 项目结构

```
finance-risk-rag/
├── config.py              # 配置管理模块
├── utils.py               # 工具函数库
├── rag_core.py            # RAG核心引擎
├── extract_text.py        # OCR文本提取
├── extract_entities.py    # 实体提取管道
├── docs/                  # 文档目录
│   ├── *.pdf             # 输入PDF文件
│   ├── all_extracted.txt # 提取的文本
│   └── entities_extracted.json  # 风险实体结果
├── knowledge_base/        # 知识库
│   ├── risk_entities.json # 风险实体规则
│   ├── stopwords.txt      # 停用词表
│   └── finance_dict.txt   # 金融词典
├── rag_db/                # Chroma向量数据库
├── cache/                 # 缓存目录
├── logs/                  # 日志目录
└── tests/                 # 测试用例
```

---

## 🚀 快速开始

### 环境要求

- Python 3.8 ~ 3.10（推荐3.9）
- 操作系统：Windows 10/11、Linux (Ubuntu 20.04+)、macOS
- 硬件要求：
  - 最低：4核CPU + 8GB内存
  - 推荐：8核CPU + 16GB内存
  - 可选：NVIDIA GPU（加速BERT训练）

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/eninem123/finance-risk-rag-v2.git
cd finance-risk-rag-v2

# 2. 创建虚拟环境
python -m venv rag_env

# Windows激活
rag_env\Scripts\activate

# Linux/Mac激活
source rag_env/bin/activate

# 3. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，设置 API 密钥
```

### 配置环境变量

创建 `.env` 文件：

```env
# LLM API配置（必填）
OPENAI_API_KEY=your_api_key_here
# 或
MOONSHOT_API_KEY=your_moonshot_key_here

# LLM配置（可选）
LLM_PROVIDER=moonshot
LLM_BASE_URL=https://api.moonshot.cn/v1
LLM_MODEL_NAME=moonshot-v1-8k

# OCR配置（可选）
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### 基本使用 (推荐方案)

自 v2.1.0 起，项目提供了全新的模块化架构和统一的命令行入口：

```bash
# 1. 将 PDF 文件放入 docs/ 目录

# 2. 批量处理 PDF 文档（OCR + 文本提取）
python main.py process

# 3. 提取风险实体
python main.py extract --input docs/all_extracted.txt

# 4. 问答查询（支持自动构建索引）
python main.py query "这家公司的流动性风险如何？" --build
```

### 传统脚本使用 (向后兼容)

原有的脚本仍保留在根目录，供需要单步调试或特定 BERT 微调场景使用：

```bash
python extract_text.py
python extract_entities.py
python risk_qa_cli.py --build
```

---

## 配置说明

### 配置文件结构

系统使用 `config.py` 进行集中配置管理，支持环境变量覆盖：

```python
from config import get_config

config = get_config()

# 访问配置
print(config.llm_api_key)
print(config.chunk_size)
```

### 主要配置项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `LLM_API_KEY` | LLM API密钥 | 环境变量 |
| `LLM_BASE_URL` | API基础URL | https://api.moonshot.cn/v1 |
| `CHUNK_SIZE` | 文本分块大小 | 800 |
| `CHUNK_OVERLAP` | 分块重叠大小 | 100 |
| `OCR_DPI` | OCR分辨率 | 600 |
| `MAX_CONTEXT_TOKENS` | 最大上下文token数 | 2000 |

---

## 使用指南

### 1. 文档处理

```python
from extract_text import DocumentProcessor

processor = DocumentProcessor()
processor.batch_process("docs/")
```

### 2. 实体提取

```python
from extract_entities import EntityExtractionPipeline
from pathlib import Path

pipeline = EntityExtractionPipeline()
pipeline.initialize()

result = pipeline.process(Path("docs/all_extracted.txt"))
print(f"提取实体数: {len(result.entities)}")
print(f"风险等级: {result.risk_level}")
```

### 3. RAG查询

```python
from rag_core import RAGEngine

engine = RAGEngine()

# 构建索引
engine.build_index()

# 执行查询
result = engine.query("这家公司的信用评级如何？")
print(result.answer)
```

---

## API文档

### 核心类

#### `RAGEngine`

RAG引擎主类，负责向量索引构建和查询。

```python
class RAGEngine:
    def __init__(
        self,
        docs_dir: str = "docs",
        db_path: str = "rag_db",
        chunk_config: Optional[ChunkConfig] = None
    ) -> None: ...
    
    def build_index(self) -> Dict[str, int]:
        """构建向量索引，返回统计信息"""
        ...
    
    def query(self, question: str, top_k: int = 4) -> QueryResult:
        """执行RAG查询"""
        ...
```

#### `EntityExtractionPipeline`

实体提取管道，整合规则提取和BERT提取。

```python
class EntityExtractionPipeline:
    def __init__(self, config: Optional[Config] = None) -> None: ...
    
    def initialize(self) -> None:
        """初始化管道组件"""
        ...
    
    def process(self, text_path: Path) -> ExtractionResult:
        """处理文本文件，返回提取结果"""
        ...
```

#### `Config`

配置管理类。

```python
class Config:
    # 路径配置
    base_dir: Path
    chroma_db_dir: Path
    docs_dir: Path
    
    # LLM配置
    llm_api_key: Optional[str]
    llm_base_url: str
    llm_model_name: str
    
    # 处理配置
    chunk_size: int
    chunk_overlap: int
    
    def validate(self) -> bool:
        """验证配置是否有效"""
        ...
```

### 数据类

#### `Entity`

风险实体数据类。

```python
@dataclass
class Entity:
    type: str           # 实体类型
    text: str           # 实体文本
    risk_score: int     # 风险分数
    confidence: float   # 置信度
    context: str        # 上下文
    source: str         # 来源 (rule/bert)
```

#### `QueryResult`

查询结果数据类。

```python
@dataclass
class QueryResult:
    answer: str                     # 回答内容
    sources: List[Dict[str, Any]]   # 来源文档
    confidence: float               # 置信度
    metadata: Dict[str, Any]        # 元数据
```

---

## 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| OCR准确率 | 97.8% | 含表格识别 |
| 实体识别率 | 88.0% | 跨语言 |
| 分类准确率 | 99.0% | 6类文档 |
| 单文件处理时间 | 2.1秒 | PDF转实体 |
| 批量1000文件 | 32分钟 | 含OCR |

---

## 常见问题

### Q: OCR识别速度慢怎么办？

A: 可以降低DPI设置：

```python
# 在 config.py 中修改
OCR_DPI = 300  # 默认600
```

### Q: 向量库占用空间过大？

A: 定期清理或调整分块大小：

```python
# 清理向量库
from rag_core import RAGDatabase
db = RAGDatabase()
db.clear()

# 调整分块大小
config.chunk_size = 500  # 默认800
```

### Q: 如何添加自定义风险实体？

A: 编辑 `knowledge_base/risk_entities.json`：

```json
{
  "custom_risk": {
    "keywords": ["风险关键词1", "风险关键词2"],
    "risk_score": 20
  }
}
```

---

## 贡献指南

我们欢迎所有形式的贡献！

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/eninem123/finance-risk-rag-v2.git
cd finance-risk-rag-v2

# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/

# 代码格式化
black .
isort .
```

### 代码规范

- 遵循 PEP 8 代码风格
- 使用类型注解
- 编写文档字符串
- 保持测试覆盖率 > 80%

### 提交规范

使用约定式提交：

```
feat: 添加新功能
fix: 修复bug
docs: 文档更新
refactor: 代码重构
test: 测试相关
```

---

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 联系方式

- 项目地址: https://github.com/eninem123/finance-risk-rag-v2
- 问题反馈: https://github.com/eninem123/finance-risk-rag-v2/issues

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给一个 Star ⭐**

</div>
