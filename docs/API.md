# Finance-Risk-RAG API 文档

## 目录

1. [核心模块](#核心模块)
2. [配置模块](#配置模块)
3. [工具模块](#工具模块)
4. [实体提取模块](#实体提取模块)
5. [数据类](#数据类)
6. [异常类](#异常类)

---

## 核心模块

### `rag_core` - RAG引擎

RAG（检索增强生成）核心引擎，负责向量索引构建和智能问答。

#### 类 `RAGEngine`

主RAG引擎类。

```python
class RAGEngine:
    def __init__(
        self,
        docs_dir: str = "docs",
        db_path: str = "rag_db",
        chunk_config: Optional[ChunkConfig] = None
    ) -> None
```

**参数:**
- `docs_dir` (str): 文档目录路径
- `db_path` (str): 向量数据库存储路径
- `chunk_config` (ChunkConfig, optional): 文本分块配置

**方法:**

##### `build_index()`

构建向量索引。

```python
def build_index(self) -> Dict[str, int]
```

**返回:**
- `Dict[str, int]`: 构建统计信息
  - `files_processed`: 处理的文件数
  - `chunks_added`: 添加的分块数
  - `errors`: 错误数

**示例:**
```python
engine = RAGEngine()
stats = engine.build_index()
print(f"处理文件: {stats['files_processed']}")
```

##### `query()`

执行RAG查询。

```python
def query(
    self,
    question: str,
    top_k: int = 4
) -> QueryResult
```

**参数:**
- `question` (str): 用户问题
- `top_k` (int): 检索文档数量，默认4

**返回:**
- `QueryResult`: 查询结果对象

**示例:**
```python
result = engine.query("这家公司的信用评级如何？")
print(result.answer)
for source in result.sources:
    print(f"来源: {source['source']}")
```

---

#### 类 `RAGDatabase`

向量数据库封装类。

```python
class RAGDatabase:
    def __init__(
        self,
        db_path: str = "rag_db",
        embedding_fn: Optional[Callable] = None
    ) -> None
```

**方法:**

| 方法 | 说明 | 参数 | 返回值 |
|------|------|------|--------|
| `add_documents()` | 添加文档到数据库 | `chunks: List[DocumentChunk]`, `batch_size: int` | `int` 添加数量 |
| `query()` | 查询相似文档 | `query_text: str`, `top_k: int` | `List[Dict]` |
| `clear()` | 清空数据库 | - | `None` |

---

#### 类 `LLMClientWrapper`

LLM客户端封装类。

```python
class LLMClientWrapper:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "moonshot-v1-8k"
    ) -> None
```

**方法:**

##### `ask()`

向LLM提问。

```python
def ask(
    self,
    query: str,
    context: str,
    temperature: float = 0.0,
    max_tokens: int = 512
) -> str
```

**参数:**
- `query` (str): 用户问题
- `context` (str): 上下文内容
- `temperature` (float): 温度参数，默认0.0
- `max_tokens` (int): 最大token数，默认512

**返回:**
- `str`: LLM回答

---

#### 类 `TextChunker`

文本分块器。

```python
class TextChunker:
    def __init__(self, config: Optional[ChunkConfig] = None) -> None
```

**方法:**

##### `chunk()`

将文本分块。

```python
def chunk(self, text: str) -> List[str]
```

**参数:**
- `text` (str): 输入文本

**返回:**
- `List[str]`: 分块后的文本列表

---

## 配置模块

### `config` - 配置管理

集中管理系统配置参数。

#### 函数 `get_config()`

获取全局配置实例。

```python
def get_config() -> Config
```

**返回:**
- `Config`: 配置对象

**示例:**
```python
from config import get_config

config = get_config()
print(config.llm_api_key)
print(config.chunk_size)
```

---

#### 类 `Config`

配置数据类。

```python
@dataclass
class Config:
    # 路径配置
    base_dir: Path
    chroma_db_dir: Path
    docs_dir: Path
    cache_dir: Path
    log_dir: Path
    knowledge_base_dir: Path
    bert_local_path: Optional[Path]
    
    # LLM配置
    llm_provider: str
    llm_api_key: Optional[str]
    llm_base_url: str
    llm_model_name: str
    max_context_tokens: int
    
    # 嵌入模型配置
    embedding_backend: str
    
    # OCR配置
    tesseract_cmd: Optional[str]
    ocr_languages: str
    ocr_dpi: int
    ocr_version: str
    
    # 风险评估配置
    risk_level_low: int
    risk_level_medium: int
    risk_level_high: int
    
    # 处理配置
    chunk_size: int
    chunk_overlap: int
    batch_size: int
    api_call_interval: float
```

**方法:**

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `ensure_directories()` | 确保所有必要目录存在 | `None` |
| `validate()` | 验证配置是否有效 | `bool` |
| `to_dict()` | 转换为字典 | `dict` |

**属性:**

| 属性 | 说明 | 类型 |
|------|------|------|
| `risk_entities_path` | 风险实体规则文件路径 | `Path` |
| `stopwords_path` | 停用词文件路径 | `Path` |
| `finance_dict_path` | 金融词典文件路径 | `Path` |
| `processing_log_path` | 处理日志文件路径 | `Path` |

---

## 工具模块

### `utils` - 工具函数

提供通用的工具函数。

#### 路径管理

| 函数 | 说明 | 参数 | 返回值 |
|------|------|------|--------|
| `ensure_dirs(*dirs)` | 确保目录存在 | `*dirs: PathLike` | `None` |
| `get_project_root()` | 获取项目根目录 | - | `Path` |
| `normalize_path(relative_path)` | 相对路径转绝对路径 | `relative_path: PathLike` | `Path` |
| `safe_delete_directory(dir_path)` | 安全删除目录 | `dir_path: PathLike` | `bool` |
| `get_file_hash(file_path)` | 计算文件哈希 | `file_path: PathLike` | `str` |

#### 文本处理

| 函数 | 说明 | 参数 | 返回值 |
|------|------|------|--------|
| `clean_text(text)` | 清洗文本 | `text: str` | `str` |
| `split_text_by_sentence(text, max_len)` | 按句子拆分 | `text: str`, `max_len: int` | `List[str]` |
| `extract_keywords(text, top_n)` | 提取关键词 | `text: str`, `top_n: int` | `List[str]` |

#### 文件操作

| 函数 | 说明 | 参数 | 返回值 |
|------|------|------|--------|
| `load_json_file(file_path)` | 加载JSON文件 | `file_path: PathLike` | `Any` |
| `save_json_file(data, file_path)` | 保存JSON文件 | `data: Any`, `file_path: PathLike` | `bool` |

#### 风险计算

| 函数 | 说明 | 参数 | 返回值 |
|------|------|------|--------|
| `calculate_risk_level(score)` | 计算风险等级 | `score: float` | `str` |
| `normalize_risk_scores(scores)` | 归一化风险分数 | `scores: List[float]` | `List[float]` |
| `calculate_risk_trend(historical_scores)` | 计算风险趋势 | `historical_scores: List[float]` | `Dict` |

#### 日志配置

```python
def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None
) -> logging.Logger
```

**示例:**
```python
from utils import setup_logger

logger = setup_logger("my_module", "logs/my_module.log")
logger.info("这是一条日志")
```

---

## 实体提取模块

### `extract_entities` - 实体提取

从财务文档中提取风险实体。

#### 类 `EntityExtractionPipeline`

实体提取管道主类。

```python
class EntityExtractionPipeline:
    def __init__(self, config: Optional[Config] = None) -> None
```

**方法:**

##### `initialize()`

初始化管道组件。

```python
def initialize(self) -> None
```

##### `process()`

处理文本文件。

```python
def process(self, text_path: Path) -> ExtractionResult
```

**参数:**
- `text_path` (Path): 文本文件路径

**返回:**
- `ExtractionResult`: 提取结果对象

**示例:**
```python
pipeline = EntityExtractionPipeline()
pipeline.initialize()
result = pipeline.process(Path("docs/all_extracted.txt"))
print(f"实体数: {len(result.entities)}")
```

##### `save_result()`

保存提取结果。

```python
def save_result(self, result: ExtractionResult, output_path: Path) -> None
```

##### `interactive_qa()`

交互式问答。

```python
def interactive_qa(self, entities: List[Entity]) -> None
```

---

#### 类 `RuleBasedExtractor`

基于规则的实体提取器。

```python
class RuleBasedExtractor:
    def __init__(self, rules_path: Optional[Path] = None) -> None
```

**方法:**

| 方法 | 说明 | 参数 | 返回值 |
|------|------|------|--------|
| `load_rules(rules_path)` | 加载实体规则 | `rules_path: Path` | `None` |
| `extract(text)` | 提取实体 | `text: str` | `List[Entity]` |

---

#### 类 `BERTExtractor`

基于BERT的实体提取器。

```python
class BERTExtractor:
    def __init__(self, model_path: Optional[Path] = None) -> None
```

**方法:**

| 方法 | 说明 | 参数 | 返回值 |
|------|------|------|--------|
| `load_model(model_path)` | 加载BERT模型 | `model_path: Path` | `bool` |
| `extract(text, chunk_size, overlap)` | 提取实体 | `text: str`, ... | `List[Entity]` |

**属性:**

| 属性 | 说明 | 类型 |
|------|------|------|
| `is_available` | 检查模型是否可用 | `bool` |

---

#### 类 `EntityMerger`

实体融合器。

```python
class EntityMerger:
    def merge(
        self,
        rule_entities: List[Entity],
        bert_entities: List[Entity]
    ) -> List[Entity]
```

---

#### 类 `RAGQAService`

RAG问答服务。

```python
class RAGQAService:
    def query(
        self,
        question: str,
        context_entities: List[Entity],
        max_tokens: int = 500
    ) -> str
```

---

## 数据类

### `Entity`

风险实体数据类。

```python
@dataclass
class Entity:
    type: str               # 实体类型
    text: str               # 实体文本
    risk_score: int         # 风险分数
    confidence: float       # 置信度
    context: str = ""       # 上下文
    source: str = "rule"    # 来源 (rule/bert)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**方法:**

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `to_dict()` | 转换为字典 | `Dict[str, Any]` |
| `key` | 实体唯一键 | `Tuple[str, str]` |

---

### `ExtractionResult`

提取结果数据类。

```python
@dataclass
class ExtractionResult:
    entities: List[Entity]          # 实体列表
    total_risk_score: int           # 总风险分数
    risk_level: str                 # 风险等级
    extraction_time: str            # 提取时间
    metadata: Dict[str, Any]        # 元数据
```

**方法:**

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `to_dict()` | 转换为字典 | `Dict[str, Any]` |

---

### `QueryResult`

查询结果数据类。

```python
@dataclass
class QueryResult:
    answer: str                     # 回答内容
    sources: List[Dict[str, Any]]   # 来源文档
    confidence: float = 1.0         # 置信度
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

### `ChunkConfig`

文本分块配置。

```python
@dataclass
class ChunkConfig:
    chunk_size: int = 800   # 分块大小
    overlap: int = 100      # 重叠大小
```

---

### `DocumentChunk`

文档分块数据类。

```python
@dataclass
class DocumentChunk:
    content: str                    # 内容
    source: str                     # 来源文件
    chunk_index: int                # 分块索引
    metadata: Dict[str, Any]        # 元数据
```

---

## 异常类

### 异常层次结构

```
Exception
├── RAGError
│   ├── EmbeddingError
│   ├── LLMError
│   └── DatabaseError
├── ExtractionError
│   └── RuleLoadError
└── UtilsError
    └── FileOperationError
```

### `RAGError`

RAG系统基础异常。

```python
class RAGError(Exception):
    """RAG系统基础异常"""
    pass
```

### `EmbeddingError`

嵌入模型相关异常。

```python
class EmbeddingError(RAGError):
    """嵌入模型相关异常"""
    pass
```

### `LLMError`

LLM调用相关异常。

```python
class LLMError(RAGError):
    """LLM调用相关异常"""
    pass
```

### `DatabaseError`

数据库相关异常。

```python
class DatabaseError(RAGError):
    """数据库相关异常"""
    pass
```

### `ExtractionError`

实体提取异常。

```python
class ExtractionError(Exception):
    """实体提取异常"""
    pass
```

---

## 使用示例

### 完整工作流示例

```python
from pathlib import Path
from config import get_config
from rag_core import RAGEngine
from extract_entities import EntityExtractionPipeline

# 1. 加载配置
config = get_config()
config.ensure_directories()

# 2. 初始化实体提取管道
pipeline = EntityExtractionPipeline(config)
pipeline.initialize()

# 3. 处理文档
result = pipeline.process(Path("docs/all_extracted.txt"))

# 4. 保存结果
pipeline.save_result(result, Path("docs/entities_extracted.json"))

# 5. 构建RAG索引
engine = RAGEngine(
    docs_dir=str(config.docs_dir),
    db_path=str(config.chroma_db_dir)
)
stats = engine.build_index()

# 6. 执行查询
query_result = engine.query("这家公司的流动性风险如何？")
print(query_result.answer)
```

### 自定义配置示例

```python
from config import Config
from rag_core import RAGEngine, ChunkConfig

# 创建自定义配置
config = Config()
config.chunk_size = 500
config.chunk_overlap = 50

# 使用自定义配置
chunk_config = ChunkConfig(
    chunk_size=config.chunk_size,
    overlap=config.chunk_overlap
)

engine = RAGEngine(chunk_config=chunk_config)
```

### 异常处理示例

```python
from rag_core import RAGEngine, RAGError, LLMError

try:
    engine = RAGEngine()
    result = engine.query("问题")
except LLMError as e:
    print(f"LLM调用失败: {e}")
except RAGError as e:
    print(f"RAG系统错误: {e}")
```
