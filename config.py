# D:\finance-risk-rag\config.py
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 本地 BERT 模型路径（已放在 D:\finance-risk-rag\hfl\chinese-bert-wwm-ext）
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
BERT_LOCAL_PATH = os.path.join(BASE_DIR, "hfl", "chinese-bert-wwm-ext")

# Chroma DB 设置
CHROMA_DB_DIR = os.path.join(BASE_DIR, "rag_db")  # 这里是 chroma 的持久化路径

# 嵌入模型说明：优先使用本地 ONNX 的 MiniLM（如果 chroma 的 ONNX 接口可用）
# 如果 ONNX 不可用，代码会 fallback 到 sentence-transformers 的 all-MiniLM-L6-v2
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "onnx_or_sbert")
# 用于限制上下文大小的参数等
MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", 2000))

# LLM 配置：我们使用外部 API（Moonshot/OpenAI）
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "moonshot")  # "moonshot" or "openai"
LLM_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.moonshot.cn/v1")

# Tesseract 可执行文件路径（如果没有加入 PATH，请修改为你的实际路径）
# 例如： r"D:\finance-risk-rag\tesseract-ocr-w64-setup-5.5.0.20241111\tesseract.exe"
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# 其他配置可在这里添加
