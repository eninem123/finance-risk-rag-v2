"""
Finance-Risk-RAG 工具模块
========================
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, List, Optional, Union

from .config import get_config

PathLike = Union[str, Path]


def ensure_dirs(*dirs: PathLike) -> None:
    for dir_path in dirs:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)


def get_file_hash(file_path: PathLike) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_text(text: str) -> str:
    """
    针对财务文本优化的清洗函数。
    处理特殊的 Unicode 字符、多余空白、并标准化标点符号。
    """
    if not text:
        return ""

    # 替换常见的乱码或特殊空白
    text = text.replace("\xa0", " ").replace("\u3000", " ")

    # 规范化标点
    text = text.replace("，", ",").replace("；", ";").replace("：", ":")
    text = text.replace("（", "(").replace("）", ")").replace("【", "[").replace("】", "]")

    # 处理数字中的中文句号（常见于 OCR 错误）
    text = re.sub(r"(?<=\d)。(?=\d)", ".", text)

    # 压缩空白
    text = re.sub(r"\s+", " ", text).strip()

    # 过滤掉不可见字符
    text = "".join(ch for ch in text if ch.isprintable())

    return text


def split_text_by_sentence(text: str, max_len: int = 400, min_len: int = 50) -> List[str]:
    """
    将文本拆分为语义完整的块，针对财务报告进行了优化。
    """
    if not text:
        return []

    # 改进的句子分隔符，更好地处理中英文混排
    sentence_seps = r"([。！？；.!?;])(?![0-9])"

    # 使用捕获分组保留分隔符
    raw_parts = re.split(sentence_seps, text)

    # 重组句子
    sentences = []
    for i in range(0, len(raw_parts) - 1, 2):
        sentences.append(raw_parts[i] + raw_parts[i + 1])
    if len(raw_parts) % 2 == 1 and raw_parts[-1]:
        sentences.append(raw_parts[-1])

    # 智能合并，确保分块不会太碎且不超过 max_len
    chunks: List[str] = []
    current_chunk = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        if len(current_chunk) + len(sent) <= max_len:
            current_chunk += (" " if current_chunk else "") + sent
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # 如果单句就超过 max_len，强制截断（虽然罕见）
            if len(sent) > max_len:
                for i in range(0, len(sent), max_len):
                    chunks.append(sent[i : i + max_len])
                current_chunk = ""
            else:
                current_chunk = sent

    if current_chunk:
        # 如果最后一个块太短，尝试合并到上一个块（如果可能）
        if (
            chunks
            and len(current_chunk) < min_len
            and len(chunks[-1]) + len(current_chunk) <= max_len
        ):
            chunks[-1] += " " + current_chunk
        else:
            chunks.append(current_chunk)

    return chunks


def load_json_file(file_path: PathLike, default: Any = None) -> Any:
    path = Path(file_path)
    if not path.exists():
        return default if default is not None else {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def save_json_file(data: Any, file_path: PathLike) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def setup_logger(
    name: str, log_file: Optional[PathLike] = None, level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


def calculate_risk_level(score: float) -> str:
    config = get_config()
    if score < config.risk_level_low:
        return "低风险"
    elif score < config.risk_level_medium:
        return "中风险"
    elif score < config.risk_level_high:
        return "高风险"
    else:
        return "极高风险"
