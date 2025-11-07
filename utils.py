# utils.py
# 银行风控RAG系统通用工具函数库（跨模块复用）
import os
import re
import logging
import json
from typing import List, Dict, Optional
import jieba
import numpy as np
# utils.py (新增函数)
import shutil
import time


def safe_delete_rag_db():
    """安全删除 rag_db 目录，解决 WinError 32"""
    if os.path.exists("rag_db"):
        print("检测到旧向量库，尝试安全删除...")
        for _ in range(5):
            try:
                shutil.rmtree("rag_db")
                print("旧向量库 rag_db/ 已安全删除")
                time.sleep(1)
                return
            except PermissionError:
                print("文件被占用，等待 2 秒后重试...")
                time.sleep(2)
            except Exception as e:
                print(f"删除失败: {e}")
                time.sleep(1)
        print("删除失败，请手动关闭占用 rag_db 的程序")
# ====================== 路径管理工具 ======================
def ensure_dirs(*dirs: str) -> None:
    """确保目录存在，不存在则创建"""
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"创建目录：{dir_path}")

def get_project_root() -> str:
    """获取项目根目录（基于当前文件路径推导）"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def normalize_path(relative_path: str) -> str:
    """将相对路径转换为项目根目录下的绝对路径"""
    return os.path.abspath(os.path.join(get_project_root(), relative_path))

# ====================== 文本处理工具 ======================
def clean_text(text: str) -> str:
    if not text:
        return ""
    # 1. 去除连续空格和换行
    text = re.sub(r'\s+', ' ', text).strip()
    # 2. 去除特殊控制字符（保留中英文标点）
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    # 3. 关键修复：分两步处理中文句号“。”
    # 3.1 先处理“数字间的中文句号”（如“3。5”→“3.5”）
    text = re.sub(r'(?<=\d)。(?=\d)', '.', text)  # 数字+中文句号+数字 → 数字.数字
    # 3.2 再处理“非数字间的中文句号”（如“风险。”→“风险.”）
    text = re.sub(r'(?<!\d)。(?!\d)', '.', text)  # 非数字+中文句号+非数字 → 英文句号
    # 4. 统一其他中文标点（逗号、分号）
    text = text.replace('，', ',').replace('；', ';')
    return text

def split_text_by_sentence(text: str, max_len: int = 200) -> List[str]:
    """按句子拆分文本（精准处理数字小数点，避免过度合并）"""
    # 1. 关键修复：正则仅匹配“非数字间的分隔符”（排除3.5中的“.”）
    # 正则解释：(?<!\d) 左边不是数字；(?![0-9.]) 右边不是数字/小数点（避免拆分3.5.6这类特殊格式）
    sentence_seps = r'(?<!\d)([。！？；.!?;])(?![0-9.])'
    # 拆分“句子内容+分隔符”，过滤空字符串
    parts = [p.strip() for p in re.split(sentence_seps, text) if p.strip()]
    
    # 2. 重组完整句子（内容+原分隔符）
    sentences = []
    i = 0
    while i < len(parts):
        # 内容部分（如“某银行2024年报告显示...”）
        content = parts[i]
        # 分隔符部分（下一个元素，如“！”“.”）
        sep = parts[i+1] if (i+1 < len(parts) and parts[i+1] in "。！？；.!?;") else "."
        sentences.append(f"{content}{sep}")
        i += 2  # 跳过已处理的“内容+分隔符”
    
    # 3. 合并过短句子（仅合并“同一话题”的短句子，避免跨话题合并）
    merged = []
    current = ""
    for sent in sentences:
        # 定义“跨话题标志”：句子以这些词开头，说明是新话题，不合并
        new_topic_flags = ["涉及", "此外", "同时", "另外", "其中", "值得注意的是", "需要说明的是"]
        is_new_topic = any(sent.startswith(flag) for flag in new_topic_flags)
        
        # 合并规则：① 不是新话题 ② 合并后总长度≤max_len ③ 当前句子非空
        if not is_new_topic and len(current) + len(sent) <= max_len and current:
            # 去掉current末尾的旧分隔符，拼接新句子（保留新句子的分隔符）
            current = current[:-1] + sent
        else:
            if current:
                merged.append(current)
            current = sent  # 新话题或长度超了，重新开始累计
    
    # 加入最后一个句子
    if current:
        merged.append(current)
    
    # 4. 最终清洗：去除空句子，修复可能的标点重复（如“！。”→“！”）
    merged_clean = []
    for s in merged:
        # 去除末尾重复的分隔符（如“120亿元！。”→“120亿元！”）
        s = re.sub(r'([。！？；.!?;])+', r'\1', s.strip())
        if s:
            merged_clean.append(s)
    
    return merged_clean

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    if not text:
        return []
    # 新增：加载金融领域自定义词典
    finance_dict_path = normalize_path("knowledge_base/finance_dict.txt")
    if os.path.exists(finance_dict_path):
        jieba.load_userdict(finance_dict_path)
    # 原有逻辑：加载停用词
    stopwords = load_stopwords()
    finance_stopwords = {"显示", "涉及", "去年", "今年", "报告", "数据", "情况", "分析", "指出", "认为"}
    stopwords.update(finance_stopwords)
    # 分词过滤
    words = jieba.cut(text)
    filtered = [w for w in words if w.strip() and w not in stopwords and len(w) > 1]
    # 词频统计与排序
    word_counts = {}
    for word in filtered:
        word_counts[word] = word_counts.get(word, 0) + 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]

# ====================== 数据缓存工具 ======================
def load_json_file(file_path: str) -> Dict:
    """安全加载JSON文件（含错误处理）"""
    if not os.path.exists(file_path):
        logging.warning(f"JSON文件不存在：{file_path}")
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.error(f"JSON文件格式错误：{file_path}")
        return {}
    except Exception as e:
        logging.error(f"加载JSON失败 {file_path}：{str(e)}")
        return {}

def save_json_file(data: Dict, file_path: str) -> bool:
    """安全保存JSON文件（含目录创建）"""
    try:
        ensure_dirs(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logging.error(f"保存JSON失败 {file_path}：{str(e)}")
        return False

# ====================== 停用词管理 ======================
def load_stopwords() -> set:
    """加载中文停用词表（适配金融文本）"""
    stopwords_path = normalize_path("knowledge_base/stopwords.txt")
    if not os.path.exists(stopwords_path):
        # 生成默认停用词表
        default_stopwords = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}
        save_json_file(list(default_stopwords), stopwords_path)
        return default_stopwords
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            return set([line.strip() for line in f if line.strip()])
    except Exception as e:
        logging.error(f"加载停用词失败：{str(e)}")
        return set()

# ====================== 风险计算工具 ======================
def calculate_risk_level(score: float) -> str:
    """根据风险总分计算风险等级"""
    if score < 30:
        return "低风险"
    elif 30 <= score < 60:
        return "中风险"
    elif 60 <= score < 90:
        return "高风险"
    else:
        return "极高风险"

def normalize_risk_scores(scores: List[float]) -> List[float]:
    """归一化风险分数到0-100区间"""
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [50.0 for _ in scores]  # 所有分数相同时默认中间值
    return [(s - min_score) / (max_score - min_score) * 100 for s in scores]

# ====================== 日志配置工具 ======================
def setup_logger(name: str, log_file: str = "app.log", level: int = logging.INFO) -> logging.Logger:
    """配置自定义日志器（按模块区分日志），自动创建目录"""
    
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)  # 关键：自动创建目录

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

if __name__ == "__main__":
    # 工具函数测试
    test_text = "   某银行2024年报告显示，流动性风险敞口达460亿元，较去年增加120亿元！\n\n  涉及关联交易金额3.5亿美元。"
    print("清洗后文本：", clean_text(test_text))
    print("句子拆分：", split_text_by_sentence(test_text))
    print("关键词提取：", extract_keywords(test_text))
    print("风险等级计算（75分）：", calculate_risk_level(75))