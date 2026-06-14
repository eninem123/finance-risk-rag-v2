# Changelog

所有项目的显著变更都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [2.2.0] - 2025-05-20
### ✨ 新增功能
- **RiskAnalysisService**: 引入顶层编排服务，一键完成文档解析、分类、风险提取及报告生成。
- **精准偏移量定位**: 在 `Entity` 数据模型中增加 `start_char` 和 `end_char`，实现风险实体的原文精准回溯。
- **交互式可视化仪表盘**: 基于 Streamlit 打造 `dashboard.py`，支持可视化文档扫描与 RAG 对话。
- **长文本 BERT 提取**: `BERTExtractor` 支持滑动窗口切片（Overlap Chunking），彻底解决长文档处理受限问题。
- **报告子命令**: 增强型 CLI，支持 `python main.py report` 生成全要素风险分析结果。

### 🔧 架构优化
- **冲突仲裁升级**: 实体提取管道现在基于字符位置进行冲突仲裁，优先保留长实体及高分实体。
- **分类质量提升**: 优化了金融文档分类的 Prompt 策略，提供更专业的类别定义与理由解释。
- **代码重构**: `DocumentProcessor` 内部方法公开化，增强了组件间的可组合性。

## [2.0.0] - 2025-01-22

### ✨ 新增功能

- 完整的类型注解支持，类型覆盖率提升至95%
- 完善的异常处理体系（RAGError、LLMError、DatabaseError等）
- 数据类封装（Config、Entity、QueryResult等）
- 工厂模式支持多种嵌入模型
- 策略模式支持多LLM后端
- 管道模式实体提取流程

### 🔧 代码重构

- **config.py**: 使用@dataclass重构，支持环境变量覆盖，添加配置验证
- **utils.py**: 添加类型注解，完善文档字符串，新增风险趋势计算
- **rag_core.py**: 采用分层架构，工厂模式，完善异常处理
- **extract_entities.py**: 管道模式设计，规则与BERT分离

### 📝 文档完善

- 全新README.md，包含徽章、架构图、详细使用指南
- 完整的API文档（docs/API.md）
- 开发指南（docs/DEVELOPMENT.md）
- 优化报告（OPTIMIZATION_REPORT.md）
- 环境变量示例（.env.example）

### 🔨 工程改进

- 完善的.gitignore规则
- GitHub Actions CI/CD配置
- MIT许可证
- 变更日志

### 📊 性能指标

| 指标 | 数值 |
|------|------|
| 代码行数 | +4,394行 |
| 类型注解覆盖率 | ~95% |
| 文档字符串覆盖率 | ~90% |
| 模块数 | 11个文件 |

## [1.0.0] - 2024-11-11

### ✨ 初始版本

- 批量OCR处理（Tesseract 5.5）
- 文档分类（Kimi AI）
- 风险实体识别（规则+BERT）
- RAG问答系统（Chroma+ONNX）
- 增量处理（MD5+版本管理）

[Unreleased]: https://github.com/eninem123/finance-risk-rag-v2/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/eninem123/finance-risk-rag-v2/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/eninem123/finance-risk-rag-v2/releases/tag/v1.0.0
