# Finance-Risk-RAG 开发指南

## 目录

1. [开发环境设置](#开发环境设置)
2. [代码规范](#代码规范)
3. [项目架构](#项目架构)
4. [模块详解](#模块详解)
5. [扩展开发](#扩展开发)
6. [测试指南](#测试指南)
7. [部署指南](#部署指南)

---

## 开发环境设置

### 1. 克隆仓库

```bash
git clone https://github.com/eninem123/finance-risk-rag-v2.git
cd finance-risk-rag-v2
```

### 2. 创建虚拟环境

```bash
# 使用 venv
python -m venv venv

# Windows激活
venv\Scripts\activate

# Linux/Mac激活
source venv/bin/activate
```

### 3. 安装依赖

```bash
# 安装运行时依赖
pip install -r requirements.txt

# 安装开发依赖
pip install pytest pytest-cov black isort mypy flake8
```

### 4. 配置IDE

#### VS Code 配置

创建 `.vscode/settings.json`:

```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=100"],
    "editor.formatOnSave": true,
    "python.analysis.typeCheckingMode": "basic"
}
```

#### PyCharm 配置

1. 设置 Python 解释器为虚拟环境
2. 启用 Flake8 代码检查
3. 配置 Black 作为格式化工具

---

## 代码规范

### Python 代码风格

项目遵循 PEP 8 代码风格，使用以下工具保证代码质量：

#### Black 格式化

```bash
# 格式化所有文件
black .

# 格式化单个文件
black rag_core.py

# 检查但不修改
black --check .
```

#### isort 导入排序

```bash
# 排序导入
isort .

# 检查但不修改
isort --check-only .
```

#### Flake8 代码检查

```bash
# 检查所有文件
flake8 .

# 检查单个文件
flake8 rag_core.py
```

#### MyPy 类型检查

```bash
# 类型检查
mypy .
```

### 代码风格指南

#### 1. 导入顺序

```python
# 标准库
import os
import json
from typing import List, Dict, Optional

# 第三方库
import numpy as np
from openai import OpenAI

# 本地模块
from config import get_config
from utils import clean_text
```

#### 2. 类型注解

```python
# 函数参数和返回值类型注解
def process_text(text: str, max_length: int = 100) -> List[str]:
    """处理文本并返回分块列表"""
    pass

# 类属性类型注解
class Entity:
    type: str
    risk_score: int
    confidence: float
```

#### 3. 文档字符串

使用 Google 风格的文档字符串：

```python
def calculate_risk_score(entities: List[Entity]) -> int:
    """
    计算风险总分。
    
    Args:
        entities: 风险实体列表
        
    Returns:
        风险总分
        
    Raises:
        ValueError: 当实体列表为空时
        
    Example:
        >>> entities = [Entity(type="credit", text="AA", risk_score=25, confidence=0.9)]
        >>> calculate_risk_score(entities)
        25
    """
    if not entities:
        raise ValueError("实体列表不能为空")
    return sum(e.risk_score for e in entities)
```

#### 4. 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 模块 | 小写下划线 | `rag_core.py` |
| 类 | 大驼峰 | `RAGEngine` |
| 函数 | 小写下划线 | `build_index()` |
| 变量 | 小写下划线 | `total_risk` |
| 常量 | 大写下划线 | `MAX_TOKENS` |
| 私有属性 | 单下划线前缀 | `_model` |

#### 5. 异常处理

```python
# 自定义异常
class RAGError(Exception):
    """RAG系统基础异常"""
    pass

# 使用异常
def query(self, question: str) -> QueryResult:
    try:
        result = self._database.query(question)
    except DatabaseError as e:
        logger.error(f"数据库查询失败: {e}")
        raise RAGError(f"查询失败: {e}") from e
```

---

## 项目架构

### 模块依赖关系

```
config.py (配置)
    ↓
utils.py (工具函数)
    ↓
rag_core.py (RAG引擎)
    ↓
extract_entities.py (实体提取)
```

### 设计原则

1. **单一职责**: 每个模块只负责一个功能
2. **依赖注入**: 通过构造函数注入依赖
3. **接口抽象**: 使用协议(Protocol)定义接口
4. **配置分离**: 配置与代码分离
5. **异常传播**: 使用自定义异常链

---

## 模块详解

### config.py - 配置模块

配置模块采用数据类设计，支持环境变量覆盖。

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """系统配置类"""
    llm_api_key: Optional[str] = None
    chunk_size: int = 800
    
    def __post_init__(self):
        """初始化后处理"""
        self._load_from_env()
```

**关键设计:**
- 使用 `@dataclass` 简化代码
- `__post_init__` 从环境变量加载
- `validate()` 方法验证配置

### utils.py - 工具模块

工具模块提供通用函数，遵循函数式设计。

```python
def clean_text(text: str) -> str:
    """清洗文本"""
    if not text:
        return ""
    # 处理逻辑
    return text
```

**关键设计:**
- 纯函数，无副作用
- 完善的类型注解
- 详细的文档字符串

### rag_core.py - RAG引擎

RAG引擎采用分层架构：

```
RAGEngine (主类)
    ├── TextChunker (分块器)
    ├── RAGDatabase (数据库)
    └── LLMClientWrapper (LLM客户端)
```

**关键设计:**
- 工厂模式创建嵌入函数
- 策略模式切换LLM后端
- 数据类封装结果

### extract_entities.py - 实体提取

实体提取采用管道模式：

```
EntityExtractionPipeline
    ├── RuleBasedExtractor (规则提取)
    ├── BERTExtractor (模型提取)
    ├── EntityMerger (融合器)
    └── RAGQAService (问答服务)
```

---

## 扩展开发

### 添加新的实体类型

1. 编辑 `knowledge_base/risk_entities.json`:

```json
{
  "new_risk_type": {
    "keywords": ["关键词1", "关键词2"],
    "risk_score": 15,
    "description": "新风险类型描述"
  }
}
```

2. 在代码中使用：

```python
from extract_entities import RuleBasedExtractor

extractor = RuleBasedExtractor()
extractor.load_rules(Path("knowledge_base/risk_entities.json"))
entities = extractor.extract(text)
```

### 添加新的LLM后端

1. 实现 `LLMClient` 协议：

```python
from typing import Protocol

class LLMClient(Protocol):
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        ...

class MyLLMClient:
    def chat(self, messages, **kwargs):
        # 实现您的LLM调用逻辑
        pass
```

2. 注册到工厂：

```python
class LLMClientFactory:
    @staticmethod
    def create(provider: str) -> LLMClient:
        if provider == "my_llm":
            return MyLLMClient()
        # ...
```

### 添加新的嵌入模型

1. 实现 `EmbeddingFunction` 协议：

```python
class MyEmbedding:
    def __call__(self, texts: List[str]) -> List[List[float]]:
        # 实现嵌入逻辑
        pass
```

2. 添加到工厂：

```python
class EmbeddingModelFactory:
    @staticmethod
    def create(backend: str):
        if backend == "my_embedding":
            return MyEmbedding()
        # ...
```

---

## 测试指南

### 测试结构

```
tests/
├── __init__.py
├── conftest.py          # pytest配置和fixtures
├── test_config.py       # 配置模块测试
├── test_utils.py        # 工具模块测试
├── test_rag_core.py     # RAG引擎测试
└── test_extract_entities.py  # 实体提取测试
```

### 编写测试

```python
import pytest
from utils import clean_text, calculate_risk_level

class TestTextUtils:
    """文本工具测试"""
    
    def test_clean_text_removes_extra_spaces(self):
        """测试去除多余空格"""
        text = "hello   world"
        result = clean_text(text)
        assert result == "hello world"
    
    def test_clean_text_handles_chinese_punctuation(self):
        """测试中文标点处理"""
        text = "3。5亿元"
        result = clean_text(text)
        assert result == "3.5亿元"
    
    @pytest.mark.parametrize("score,expected", [
        (25, "低风险"),
        (45, "中风险"),
        (75, "高风险"),
        (95, "极高风险"),
    ])
    def test_calculate_risk_level(self, score, expected):
        """测试风险等级计算"""
        assert calculate_risk_level(score) == expected
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_utils.py

# 运行特定测试
pytest tests/test_utils.py::TestTextUtils::test_clean_text_removes_extra_spaces

# 生成覆盖率报告
pytest --cov=. --cov-report=html
```

### 测试覆盖率目标

| 模块 | 目标覆盖率 |
|------|-----------|
| config.py | 90% |
| utils.py | 95% |
| rag_core.py | 85% |
| extract_entities.py | 80% |

---

## 部署指南

### Docker部署

1. 创建 `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 运行
CMD ["python", "rag_core.py", "--build-db"]
```

2. 构建镜像：

```bash
docker build -t finance-risk-rag:latest .
```

3. 运行容器：

```bash
docker run -d \
    -e OPENAI_API_KEY=your_key \
    -v $(pwd)/docs:/app/docs \
    -v $(pwd)/rag_db:/app/rag_db \
    finance-risk-rag:latest
```

### Kubernetes部署

1. 创建 `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: finance-risk-rag
spec:
  replicas: 2
  selector:
    matchLabels:
      app: finance-risk-rag
  template:
    metadata:
      labels:
        app: finance-risk-rag
    spec:
      containers:
      - name: app
        image: finance-risk-rag:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
        volumeMounts:
        - name: docs
          mountPath: /app/docs
        - name: rag-db
          mountPath: /app/rag_db
      volumes:
      - name: docs
        persistentVolumeClaim:
          claimName: docs-pvc
      - name: rag-db
        persistentVolumeClaim:
          claimName: rag-db-pvc
```

2. 部署：

```bash
kubectl apply -f k8s/deployment.yaml
```

### 定时任务部署

#### Linux Cron

```bash
# 编辑crontab
crontab -e

# 每小时执行一次
0 * * * * cd /path/to/finance-risk-rag && /path/to/venv/bin/python extract_entities.py
```

#### Windows 任务计划

1. 创建 `auto_process.bat`:

```batch
@echo off
cd /d "C:\path\to\finance-risk-rag"
call venv\Scripts\activate
python extract_entities.py
```

2. 在任务计划程序中添加任务

---

## 性能优化建议

### 1. 批量处理

```python
# 使用批量处理减少API调用
def batch_process(files: List[Path], batch_size: int = 10):
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        # 处理批次
```

### 2. 缓存策略

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text: str) -> List[float]:
    """缓存嵌入结果"""
    return model.encode(text)
```

### 3. 异步处理

```python
import asyncio

async def process_files(files: List[Path]):
    tasks = [process_file(f) for f in files]
    return await asyncio.gather(*tasks)
```

### 4. 内存优化

```python
# 使用生成器处理大文件
def read_large_file(path: Path):
    with open(path, 'r') as f:
        for line in f:
            yield line.strip()
```

---

## 常见问题

### Q: 如何调试？

```python
import logging

# 启用调试日志
logging.basicConfig(level=logging.DEBUG)

# 在模块中使用
logger = logging.getLogger(__name__)
logger.debug("调试信息")
```

### Q: 如何添加新功能？

1. 创建新模块或扩展现有模块
2. 添加类型注解和文档字符串
3. 编写单元测试
4. 更新API文档

### Q: 如何贡献代码？

1. Fork仓库
2. 创建功能分支
3. 提交代码
4. 创建Pull Request

---

## 联系方式

如有问题，请提交 Issue 或联系维护团队。
