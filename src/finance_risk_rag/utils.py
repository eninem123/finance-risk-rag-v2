import hashlib
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any, List, Optional, Union

from finance_risk_rag.exceptions import FileOperationError

# 类型别名
PathLike = Union[str, Path]


def ensure_dirs(*dirs: PathLike) -> None:
    """
    确保目录存在，不存在则创建
    """
    for dir_path in dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)


def safe_delete_directory(
    dir_path: PathLike, max_retries: int = 5, retry_delay: float = 2.0
) -> bool:
    """
    安全删除目录（解决Windows文件占用问题）
    """
    path = Path(dir_path)

    if not path.exists():
        return True

    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            time.sleep(1)
            return True
        except PermissionError:
            time.sleep(retry_delay)
        except Exception:
            time.sleep(1)

    return False


def get_file_hash(file_path: PathLike, algorithm: str = "md5") -> str:
    """
    计算文件哈希值
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

    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
    except Exception as e:
        raise FileOperationError(f"读取文件哈希失败: {path}, {e}")

    return hasher.hexdigest()


def clean_text(text: str) -> str:
    """
    清洗文本
    """
    if not text:
        return ""

    # 去除连续空格和换行
    text = re.sub(r"\s+", " ", text).strip()

    # 去除特殊控制字符
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)

    # 统一中文标点 (考虑小数点 3。5 -> 3.5)
    text = text.replace("。", ".").replace("，", ",").replace("；", ";")

    return text


def split_text_by_sentence(text: str, max_len: int = 200, min_len: int = 20) -> List[str]:
    """
    按句子拆分文本
    """
    if not text:
        return []

    sentence_seps = r"(?<!\d)([。！？；.!?;])(?![0-9.])"
    # 使用捕获组保留分隔符
    parts = [p.strip() for p in re.split(sentence_seps, text) if p is not None]

    sentences: List[str] = []
    i = 0
    while i < len(parts):
        content = parts[i]
        if not content and i + 1 < len(parts):  # 处理连续分隔符或起始分隔符
            i += 1
            continue

        if i + 1 < len(parts) and parts[i + 1] in "。！？；.!?;":
            sep = parts[i + 1]
            sentences.append(f"{content}{sep}")
            i += 2
        else:
            if content:
                sentences.append(content)
            i += 1

    merged: List[str] = []
    current = ""

    new_topic_flags = {
        "涉及",
        "此外",
        "同时",
        "另外",
        "其中",
        "值得注意的是",
        "需要说明的是",
        "综上所述",
    }

    for sent in sentences:
        is_new_topic = any(sent.startswith(flag) for flag in new_topic_flags)

        # 只有在明确需要分块（即设置了合理的 max_len 且当前块已存在内容）时才合并
        if not is_new_topic and current and len(current) + len(sent) <= max_len:
            current = current + sent
        else:
            if current:
                merged.append(current)
            current = sent

    if current:
        merged.append(current)

    result = []
    for s in merged:
        s = re.sub(r"([。！？；.!?;])+", r"\1", s.strip())
        if s and len(s) >= min_len:
            result.append(s)

    return result


def load_json_file(file_path: PathLike, default: Any = None) -> Any:
    """
    安全加载JSON文件
    """
    path = Path(file_path)
    if not path.exists():
        return default if default is not None else {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def save_json_file(
    data: Any, file_path: PathLike, ensure_dir: bool = True, indent: int = 2
) -> bool:
    """
    安全保存JSON文件
    """
    path = Path(file_path)
    try:
        if ensure_dir:
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        return True
    except Exception:
        return False


def calculate_risk_level(score: float) -> str:
    """
    根据风险总分计算风险等级
    """
    if score < 30:
        return "低风险"
    elif score < 60:
        return "中风险"
    elif score < 90:
        return "高风险"
    else:
        return "极高风险"


def setup_logger(
    name: str, log_file: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """
    配置自定义日志器
    """
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
