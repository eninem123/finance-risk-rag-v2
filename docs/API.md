# Finance-Risk-RAG API 文档

## 目录

1. [配置模块 (config)](#配置模块-config)
2. [文档处理模块 (processor)](#文档处理模块-processor)
3. [实体提取模块 (extractor)](#实体提取模块-extractor)
4. [RAG 引擎模块 (engine)](#rag-引擎模块-engine)
5. [LLM 客户端模块 (llm)](#llm-客户端模块-llm)
6. [工具模块 (utils)](#工具模块-utils)
7. [数据类 (models)](#数据类-models)
8. [异常类 (exceptions)](#异常类-exceptions)

---

## 配置模块 (config)

集中管理系统配置参数，支持环境变量覆盖。

#### 函数 `get_config()`
获取全局配置实例。
```python
def get_config() -> Config
```

#### 类 `Config`
系统配置类，包含路径、LLM、OCR 及风险评估配置。

---

## 文档处理模块 (processor)

负责 PDF 解析、OCR 识别及文档分类。

#### 类 `DocumentProcessor`
```python
def __init__(self, config=None, llm_client=None)
```
- **process_directory(docs_dir=None, max_workers=4)**: 并行处理目录下的 PDF。
- **extract_text_from_pdf(pdf_path)**: 从 PDF 提取文本（自动切换 OCR）。
- **classify_document(text_sample)**: 基于 LLM 进行文档分类。

---

## 实体提取模块 (extractor)

混合动力实体识别引擎。

#### 类 `EntityExtractionPipeline`
```python
def __init__(self, config=None, rule_extractor=None, bert_extractor=None)
```
- **process(text_or_path)**: 执行完整的提取与仲裁流程。
- **_merge_and_arbitrate(rule_entities, bert_entities)**: 内部方法，负责处理实体重叠与冲突。

#### 类 `RuleBasedExtractor`
基于正则与关键字的提取器。

#### 类 `BERTExtractor`
基于 Hugging Face Transformers 的深度学习提取器。

---

## RAG 引擎模块 (engine)

负责向量索引构建和检索增强问答。

#### 类 `RAGEngine`
```python
def __init__(self, config=None, llm_client=None)
```
- **build_index()**: 自动构建索引。
- **add_documents(txt_files, force=False)**: 增量添加文档到向量库（支持 Hash 校验）。
- **query(question, top_k=4)**: 执行 RAG 查询。

---

## LLM 客户端模块 (llm)

统一的 LLM 接口封装。

#### 类 `LLMClientWrapper`
```python
def __init__(self, api_key=None, base_url=None, model_name=None)
```
- **chat(messages, temperature=0.0, max_retries=3)**: 发送聊天请求，支持指数退避重试。
- **ask(query, context)**: 针对 RAG 优化的便捷问答方法。

---

## 工具模块 (utils)

提供文本清洗、分句、风险计算等辅助功能。

- **clean_text(text)**: 针对财务文档优化的清洗逻辑。
- **split_text_by_sentence(text, max_len=400, min_len=50)**: 语义感知的分块逻辑。
- **calculate_risk_level(score)**: 根据配置映射风险等级。

---

## 数据类 (models)

- **Entity**: 风险实体。
- **ExtractionResult**: 提取任务结果。
- **QueryResult**: 问答查询结果。
- **ClassificationResult**: 文档分类结果。

---

## 异常类 (exceptions)

- **FinanceRiskRAGError**: 基础异常。
- **OCRError**: OCR 相关错误。
- **LLMError**: LLM 调用错误（含重试失败）。
- **DatabaseError**: 向量数据库错误。
- **ExtractionError**: 实体提取错误。
