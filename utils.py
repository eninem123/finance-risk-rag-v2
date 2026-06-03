"""
Finance-Risk-RAG 工具模块
========================

提供通用的工具函数，包括文本处理、文件操作、日志配置等。

模块功能:
    - 路径管理工具
    - 文本清洗与分句
    - 关键词提取
    - JSON文件操作
    - 风险计算
    - 日志配置
"""

import hashlib
import json
import logging
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import jieba
import numpy as np

# 类型别名
PathLike = Union[str, Path]


# ==================== 异常定义 ====================

class UtilsError(Exception):
    """工具模块基础异常"""
    pass


class FileOperationError(UtilsError):
    """文件操作异常"""
    pass


class TextProcessingError(UtilsError):
    """文本处理异常"""
    pass


# ==================== 路径管理工具 ====================

def ensure_dirs(*dirs: PathLike) -> None:
    """
    确保目录存在，不存在则创建

    Args:
        *dirs: 目录路径列表

    Example:
        >>> ensure_dirs("logs", "cache", "output")
    """
    for dir_path in dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logging.debug(f"创建目录: {path}")


def get_project_root() -> Path:
    """
    获取项目根目录

    Returns:
        项目根目录路径
    """
    return Path(__file__).parent.resolve()


def normalize_path(relative_path: PathLike) -> Path:
    """
    将相对路径转换为绝对路径

    Args:
        relative_path: 相对路径

    Returns:
        绝对路径
    """
    return get_project_root() / relative_path


def safe_delete_directory(
    dir_path: PathLike,
    max_retries: int = 5,
    retry_delay: float = 2.0
) -> bool:
    """
    安全删除目录（解决Windows文件占用问题）

    Args:
        dir_path: 目录路径
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）

    Returns:
        是否删除成功
    """
    path = Path(dir_path)

    if not path.exists():
        return True

    print(f"检测到旧目录，尝试安全删除: {path}")

    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            print(f"目录已安全删除: {path}")
            time.sleep(1)
            return True
        except PermissionError:
            print(f"文件被占用，等待 {retry_delay} 秒后重试... (尝试 {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
        except Exception as e:
            print(f"删除失败: {e}")
            time.sleep(1)

    print(f"删除失败，请手动关闭占用 {path} 的程序")
    return False


def get_file_hash(file_path: PathLike, algorithm: str = "md5") -> str:
    """
    计算文件哈希值

    Args:
        file_path: 文件路径
        algorithm: 哈希算法 (md5, sha1, sha256)

    Returns:
        文件哈希值
    """
    path = Path(file_path)

    if algorithm == "md5":
        hasher = hashlib.md5()
    elif algorithm == "sha1":
        hasher = hashlib.sha1()
    elif algorithm == "sha256":
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"不支持的哈希算法: {algorithm}")

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


# ==================== 文本处理工具 ====================

def clean_text(text: str) -> str:
    """
    清洗文本

    处理内容:
        1. 去除连续空格和换行
        2. 去除特殊控制字符
        3. 统一中文标点为英文标点
        4. 处理数字间的中文句号

    Args:
        text: 原始文本

    Returns:
        清洗后的文本
    """
    if not text:
        return ""

    # 去除连续空格和换行
    text = re.sub(r'\s+', ' ', text).strip()

    # 去除特殊控制字符（保留中英文标点）
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)

    # 处理中文句号（分两步避免破坏小数点）
    # 1. 数字间的中文句号 -> 英文句号 (如 3。5 -> 3.5)
    text = re.sub(r'(?<=\d)。(?=\d)', '.', text)
    # 2. 非数字间的中文句号 -> 英文句号
    text = re.sub(r'(?<!\d)。(?!\d)', '.', text)

    # 统一其他中文标点
    text = text.replace('，', ',').replace('；', ';')

    return text


def split_text_by_sentence(
    text: str,
    max_len: int = 200,
    min_len: int = 20
) -> List[str]:
    """
    按句子拆分文本

    特点:
        - 精准处理数字小数点，避免过度拆分
        - 合并过短句子
        - 识别话题边界

    Args:
        text: 输入文本
        max_len: 单句最大长度
        min_len: 单句最小长度（低于此值会尝试合并）

    Returns:
        句子列表
    """
    if not text:
        return []

    # 句子分隔符正则（排除数字间的小数点）
    sentence_seps = r'(?<!\d)([。！？；.!?;])(?![0-9.])'

    # 拆分句子
    parts = [p.strip() for p in re.split(sentence_seps, text) if p.strip()]

    # 重组完整句子
    sentences: List[str] = []
    i = 0
    while i < len(parts):
        content = parts[i]
        sep = parts[i + 1] if (i + 1 < len(parts) and parts[i + 1] in "。！？；.!?;") else "."
        sentences.append(f"{content}{sep}")
        i += 2

    # 合并过短句子（避免跨话题合并）
    merged: List[str] = []
    current = ""

    # 话题边界标志词
    new_topic_flags = {
        "涉及", "此外", "同时", "另外", "其中",
        "值得注意的是", "需要说明的是", "综上所述"
    }

    for sent in sentences:
        is_new_topic = any(sent.startswith(flag) for flag in new_topic_flags)

        if not is_new_topic and len(current) + len(sent) <= max_len and current:
            # 合并句子
            current = current[:-1] + sent
        else:
            if current:
                merged.append(current)
            current = sent

    if current:
        merged.append(current)

    # 清理重复标点
    result = []
    for s in merged:
        s = re.sub(r'([。！？；.!?;])+', r'\1', s.strip())
        if s and len(s) >= min_len:
            result.append(s)

    return result


def extract_keywords(
    text: str,
    top_n: int = 10,
    min_word_len: int = 2
) -> List[str]:
    """
    提取关键词

    Args:
        text: 输入文本
        top_n: 返回关键词数量
        min_word_len: 最小词长度

    Returns:
        关键词列表
    """
    if not text:
        return []

    # 加载金融领域自定义词典
    finance_dict_path = normalize_path("knowledge_base/finance_dict.txt")
    if finance_dict_path.exists():
        jieba.load_userdict(str(finance_dict_path))

    # 加载停用词
    stopwords = load_stopwords()

    # 金融领域额外停用词
    finance_stopwords = {
        "显示", "涉及", "去年", "今年", "报告", "数据",
        "情况", "分析", "指出", "认为", "表示", "说明"
    }
    stopwords.update(finance_stopwords)

    # 分词过滤
    words = jieba.cut(text)
    filtered = [
        w for w in words
        if w.strip() and w not in stopwords and len(w) >= min_word_len
    ]

    # 词频统计
    word_counts: Dict[str, int] = {}
    for word in filtered:
        word_counts[word] = word_counts.get(word, 0) + 1

    # 排序返回
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]


# ==================== 数据缓存工具 ====================

def load_json_file(file_path: PathLike, default: Any = None) -> Any:
    """
    安全加载JSON文件

    Args:
        file_path: 文件路径
        default: 加载失败时的默认返回值

    Returns:
        JSON数据或默认值
    """
    path = Path(file_path)

    if not path.exists():
        logging.warning(f"JSON文件不存在: {path}")
        return default if default is not None else {}

    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"JSON文件格式错误: {path}, {e}")
        return default if default is not None else {}
    except Exception as e:
        logging.error(f"加载JSON失败: {path}, {e}")
        return default if default is not None else {}


def save_json_file(
    data: Any,
    file_path: PathLike,
    ensure_dir: bool = True,
    indent: int = 2
) -> bool:
    """
    安全保存JSON文件

    Args:
        data: 要保存的数据
        file_path: 文件路径
        ensure_dir: 是否自动创建目录
        indent: 缩进空格数

    Returns:
        是否保存成功
    """
    path = Path(file_path)

    try:
        if ensure_dir:
            path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

        return True
    except Exception as e:
        logging.error(f"保存JSON失败: {path}, {e}")
        return False


# ==================== 停用词管理 ====================

def load_stopwords() -> Set[str]:
    """
    加载中文停用词表

    Returns:
        停用词集合
    """
    stopwords_path = normalize_path("knowledge_base/stopwords.txt")

    if not stopwords_path.exists():
        # 生成默认停用词表
        default_stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "个"
        }
        save_json_file(list(default_stopwords), stopwords_path)
        return default_stopwords

    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            return {line.strip() for line in f if line.strip()}
    except Exception as e:
        logging.error(f"加载停用词失败: {e}")
        return set()


# ==================== 风险计算工具 ====================

def calculate_risk_level(score: float) -> str:
    """
    根据风险总分计算风险等级

    Args:
        score: 风险总分

    Returns:
        风险等级描述
    """
    if score < 30:
        return "低风险"
    elif score < 60:
        return "中风险"
    elif score < 90:
        return "高风险"
    else:
        return "极高风险"


def normalize_risk_scores(scores: List[float]) -> List[float]:
    """
    归一化风险分数到0-100区间

    Args:
        scores: 原始风险分数列表

    Returns:
        归一化后的风险分数列表
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [50.0 for _ in scores]

    return [(s - min_score) / (max_score - min_score) * 100 for s in scores]


def calculate_risk_trend(
    historical_scores: List[float],
    window_size: int = 3
) -> Dict[str, Any]:
    """
    计算风险趋势

    Args:
        historical_scores: 历史风险分数列表
        window_size: 移动平均窗口大小

    Returns:
        趋势分析结果
    """
    if len(historical_scores) < 2:
        return {
            "trend": "stable",
            "change_rate": 0.0,
            "prediction": historical_scores[-1] if historical_scores else 0.0
        }

    # 计算变化率
    changes = [
        historical_scores[i] - historical_scores[i - 1]
        for i in range(1, len(historical_scores))
    ]

    avg_change = sum(changes) / len(changes)

    # 判断趋势
    if avg_change > 5:
        trend = "rising"
    elif avg_change < -5:
        trend = "declining"
    else:
        trend = "stable"

    # 简单预测
    prediction = historical_scores[-1] + avg_change
    prediction = max(0, min(100, prediction))  # 限制在0-100范围

    return {
        "trend": trend,
        "change_rate": round(avg_change, 2),
        "prediction": round(prediction, 2)
    }


# ==================== 日志配置工具 ====================

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    配置自定义日志器

    Args:
        name: 日志器名称
        log_file: 日志文件路径（可选）
        level: 日志级别
        format_str: 日志格式字符串

    Returns:
        配置好的日志器
    """
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_str)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除已有的处理器
    logger.handlers.clear()

    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        if log_path.parent and not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# ==================== 向后兼容 ====================

def safe_delete_rag_db() -> None:
    """
    安全删除rag_db目录（向后兼容函数）

    已弃用，请使用 safe_delete_directory("rag_db")
    """
    safe_delete_directory("rag_db")


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 文本处理测试
    test_text = """
    某银行2024年报告显示，流动性风险敞口达460亿元，较去年增加120亿元！

    涉及关联交易金额3.5亿美元。此外，信用评级为AA。
    """

    print("=" * 50)
    print("文本处理测试")
    print("=" * 50)

    print("\n清洗后文本:")
    print(clean_text(test_text))

    print("\n句子拆分:")
    sentences = split_text_by_sentence(test_text)
    for i, s in enumerate(sentences, 1):
        print(f"  {i}. {s}")

    print("\n关键词提取:")
    keywords = extract_keywords(test_text)
    print(f"  {keywords}")

    print("\n风险等级计算:")
    print(f"  75分 -> {calculate_risk_level(75)}")
    print(f"  25分 -> {calculate_risk_level(25)}")
    print(f"  95分 -> {calculate_risk_level(95)}")

    print("\n风险趋势分析:")
    scores = [30, 35, 42, 50, 58, 65]
    trend = calculate_risk_trend(scores)
    print(f"  历史分数: {scores}")
    print(f"  趋势: {trend}")
