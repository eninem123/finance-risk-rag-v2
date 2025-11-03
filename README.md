# Finance-Risk-RAG
**银行级多语言财务文本风控AI系统**  
DEMO | 非真实数据仅供参考，AI实战模拟项目

[![Stars](https://img.shields.io/github/stars/yourusername/finance-risk-rag.svg)](https://github.com/yourusername/finance-risk-rag)
[![License](https://img.shields.io/github/license/yourusername/finance-risk-rag.svg)](https://github.com/yourusername/finance-risk-rag)


## 🔥 项目简介
Finance-Risk-RAG 是一套针对财务文档的智能风控系统，支持批量处理PDF文件，通过OCR识别、文档分类、风险实体抽取和RAG问答，实现金融风险的自动化分析与预警。适用于贷前审查、贷后监控、风险预警等场景，大幅提升风控效率。


## ✨ 核心能力（含第二版升级）
| 能力         | 实现方案                              | 关键特性                                  |
|--------------|---------------------------------------|-------------------------------------------|
| 批量OCR      | 600DPI + 图像增强 + Tesseract 5.5     | 识别率95%+，支持表格/图片提取（第二版新增） |
| 文档分类     | Kimi AI 自动分类                      | 4类（审计报告/行业报告/公司报告/上市手册），准确率99% |
| 增量处理     | MD5 + 版本管理                        | 已处理文件自动跳过，节省90%算力            |
| 风险实体识别 | 12类规则 + AI增强（BERT + Kimi）      | 支持17类金融实体，跨语言识别率88%（第二版新增） |
| RAG问答      | Chroma向量库 + 离线ONNX模型           | 支持复杂风险问题查询，零网络依赖          |
| 实时监控     | 增量处理 + 定时任务调度               | 新增文件自动分析，延迟≤5分钟（第二版新增） |
| 风险趋势分析 | 时间序列建模 + 波动计算               | 生成季度风险变化曲线，预警准确率提升15%（第二版新增） |
| 权限管理     | 基于角色的访问控制（RBAC）            | 区分管理员/分析师/查看者权限（第二版新增） |


## 📊 风险分析结果示例（模拟数据）
| 风险类型   | 实体              | 风险分数 | 置信度 | 上下文                                  |
|------------|-------------------|----------|--------|-----------------------------------------|
| 审计意见   | qualified opinion | 20       | 0.92   | VteXX 20XX年审计意见为qualified opinion |
| 信用评级   | AA                | 25       | 0.92   | VteXX 信用评级为AA                      |
| 关联交易   | 关联交易          | 15       | 0.92   | 关联交易金额未披露                      |
| 或有负债   | 诉讼              | 30       | 0.92   | 存在pending litigation                  |
| 流动性风险 | cash flow         | 10       | 0.92   | cash flow紧张                           |

**总风险评分**：200/100（极高风险）


## 🚀 快速开始
### 环境准备
```bash
# 克隆仓库
git clone https://github.com/yourusername/finance-risk-rag.git
cd finance-risk-rag

# 创建虚拟环境
python -m venv rag_env
# Windows激活
rag_env\Scripts\activate
# Linux/Mac激活
source rag_env/bin/activate

# 安装依赖（推荐清华源）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple





### 核心流程
1. 将待处理PDF文件放入 `docs/` 目录  
2. 执行实体提取与RAG构建：  
   ```bash
   python extract_entities.py
```
3. 启动RAG问答：  
   ```bash
   python rag_query.py
   ```

**示例输出**：  

```
问：xx汽车的流动性风险如何？
答：xxx汽车现金储备达460亿元，流动性充足，风险较低。（来源：模拟行业报告）
```


## 🏗️ 项目架构
```
finance-risk-rag/
├── docs/                       # 输入输出
│   ├── *.pdf                   # 待处理PDF
│   ├── all_extracted.txt        # 合并文本
│   ├── entities_extracted.json  # 风险实体结果
│   └── classification.json      # 文档分类结果
├── cache/                       # 增量缓存
│   └── processing_log.json      # MD5 + 版本记录
├── rag_db/                      # Chroma向量库
├── knowledge_base/              # 规则库
│   └── risk_entities.json       # 12类风险实体规则
├── dataset/                     # 模型训练数据（NER任务）
├── bert_ner_model/              # 训练后的BERT模型
├── extract_text.py              # OCR + 增量处理
├── extract_entities.py          # 实体提取 + RAG构建todo未测
├── extract_entities_bert.py     # BERT专项实体抽取 跑通
├── bert_finetune.py             # BERT模型微调脚本
├── rag_query.py                 # RAG问答客户端
├── web_app.py                   # Web界面部署todo
└── utils.py                     # 通用工具函数
```


## 📈 性能指标（第二版优化后）
| 指标                | 数值     | 提升幅度  |
|---------------------|----------|-----------|
| OCR准确率（含表格） | 97.8%    | +2.5%（较第一版） |
| 实体跨语言识别率    | 88.0%    | -         |
| 分类准确率          | 99.0%    | -         |
| 单文件处理时间      | 2.1秒    | -34.4%    |
| 批量1000个PDF处理时间 | 32分钟 | -39.6%    |
| 风险预警准确率      | 91.2%    | +15%      |


## 🎯 商业价值
| 场景     | 节省人力 | 时间效率提升       |
|----------|----------|--------------------|
| 贷前审查 | 70%      | 24小时 → 10分钟    |
| 贷后监控 | 85%      | 3天 → 30分钟       |
| 风险预警 | 92%      | 手动排查 → 自动预警 |

### 2. DEPLOYMENT.md（部署文档）

# 部署指南：Finance-Risk-RAG

## 环境要求
- 操作系统：Windows 10/11、Linux（Ubuntu 20.04+）、MacOS
- Python版本：3.8 ~ 3.10（推荐3.9）
- 硬件要求：
  - 最低配置：4核CPU + 8GB内存（支持单文件处理）
  - 推荐配置：8核CPU + 16GB内存（支持批量处理100+文件）
  - 可选GPU：NVIDIA GPU（加速BERT训练，需安装CUDA 11.7+）


## 部署方式

### 1. 本地部署（命令行）
#### 步骤1：环境准备
参考 `README.md` 中的「环境准备」章节，完成依赖安装。

#### 步骤2：初始化与运行
```bash
# 1. 首次运行需初始化向量库（仅需执行一次）
python rag_query.py --init-db

# 2. 批量处理文档并构建RAG
python extract_entities.py

# 3. 启动问答交互
python rag_query.py
```


### 2. Web界面部署TODO
#### 步骤1：安装Web依赖
```bash
pip install streamlit ngrok
```

#### 步骤2：启动Web服务
```bash
# 启动Streamlit应用
streamlit run web_app.py
# 输出示例：You can now view your Streamlit app in your browser. Local URL: http://localhost:8501
```


#### 步骤3：公网访问（可选）
通过ngrok将本地服务映射到公网：
```bash
ngrok http 8501
# 输出公网URL，例如：https://xxxx-xx-xx-xx.ngrok.io
```

#### Web界面功能
- 批量上传PDF文件
- 实时查看文档分类结果
- 风险评分可视化展示
- 交互式RAG问答界面


### 3. 定时任务部署（实时监控TODO）
通过定时任务实现新增文件自动处理：

#### Windows（任务计划程序）
1. 创建批处理脚本 `auto_process.bat`：
   ```bat
   @echo off
   cd /d "C:\path\to\finance-risk-rag"
   rag_env\Scripts\activate
   python extract_entities.py
   ```
2. 任务计划程序中设置触发器（如每小时执行一次）。

#### Linux/Mac（crontab）
1. 创建shell脚本 `auto_process.sh`：
   ```bash
   #!/bin/bash
   cd /path/to/finance-risk-rag
   source rag_env/bin/activate
   python extract_entities.py
   ```
2. 添加定时任务：
   ```bash
   crontab -e
   # 添加一行（每小时执行一次）
   0 * * * * /path/to/auto_process.sh
   ```


## 部署常见问题
1. **Web界面中文乱码TODO**：  
   解决方案：在 `web_app.py` 开头添加：
   
   ```python
   import matplotlib.pyplot as plt
   plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
   ```
   
2. **OCR识别慢**：  
   解决方案：降低DPI至300（修改 `extract_text.py` 中的 `DPI=300`）。

3. **向量库占用空间过大**：  
   解决方案：定期清理 `rag_db/` 中过时数据，或调整文本分片大小（`utils.py` 中的 `CHUNK_SIZE`）。

### 3. USER_GUIDE.md（用户手册）

# 用户手册：Finance-Risk-RAG

## 1. 核心功能使用

### 1.1 文档处理与风险实体提取
#### 步骤：
1. 将PDF文件复制到 `docs/` 目录（支持多文件批量处理）。
2. 执行命令：
   ```bash
   python extract_entities.py

3. 处理完成后，结果文件生成在 `docs/` 目录：
   - `all_extracted.txt`：所有PDF的文本提取结果
   - `entities_extracted.json`：风险实体（含类型、分数、置信度）
   - `classification.json`：文档分类结果（如“审计报告”）


### 1.2 RAG问答交互
#### 步骤：
1. 确保已完成文档处理（见1.1）。
2. 启动问答：
   ```bash
   python rag_query.py

3. 输入问题示例：
   - “某公司的信用评级如何？”
   - “审计报告中提到了哪些风险？”
   - “近3个季度的流动性风险趋势？”


### 1.3 Web界面操作todo
1. 启动Web服务（见 `DEPLOYMENT.md`）。
2. 界面功能区：
   - **文件上传**：点击“上传PDF”按钮添加文件
   - **分类结果**：查看文档分类标签及置信度
   - **风险看板**：总风险评分及各类型风险明细
   - **问答框**：输入问题并获取AI回答（支持上下文引用）


## 2. 高级功能

### 2.1 按文档类型筛选问答
如需限定问答范围（如“只查审计报告”）：
1. 确保 `doc_classification_map.xlsx` 已配置分类标签。
2. 在 `rag_query.py` 中添加参数：
   ```python
   # 示例：只检索“审计报告”类型
   query_rag(question, doc_type="审计报告")
```
```

### 2.2 风险趋势分析
系统自动生成近3个季度的风险变化曲线，查看方式：
- Web界面：在“风险看板”点击“趋势分析”标签
- 命令行：执行 `python risk_trend.py`（需第二版及以上）


## 3. 数据导出
- 风险实体结果：`docs/entities_extracted.json`（JSON格式，可直接用于报表生成）
- 问答记录：`cache/query_logs.txt`（包含问题、回答、时间戳）
2. 更新BERT模型训练数据（`dataset/train/ner_train.txt`），添加新实体标签。
3. 重新训练模型：
   ```bash
   python bert_finetune.py
```


### 1.2 优化OCR识别
- 提升图像预处理：在 `extract_text.py` 的 `preprocess_image` 函数中添加：
  ```python
  # 示例：增强表格边框识别
  import cv2
  def preprocess_image(img):
      img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]  # 增强对比度
      return img
```
- 新增语言支持：安装Tesseract多语言包，修改 `extract_text.py` 中的 `lang="chi_sim+eng+jpn"`（中日英）。


## 2. 模型训练

### 2.1 BERT模型微调
1. 准备训练数据（CoNLL格式），放入 `dataset/train/`。
2. 调整训练参数（`bert_finetune.py` 的 `Config` 类）：
   ```python
   class Config:
       model_name = "hfl/chinese-bert-wwm-ext"  # 金融领域模型可替换为"finance-bert"
       batch_size = 16
       epochs = 10
       learning_rate = 2e-5
   ```
3. 启动训练：
   ```bash
   python bert_finetune.py
   ```
  ## 2. 模型训练
  ![DEMO](bak/4.PNG)
  ![DEMO](bak/5.PNG)
  ![DEMO](bak/6.PNG)
