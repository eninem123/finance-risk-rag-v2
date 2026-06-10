"""
Finance-Risk-RAG 工具模块
"""

import hashlib
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any, List, Optional, Union

PathLike = Union[str, Path]


def ensure_dirs(*dirs: PathLike) -> None:
    for dir_path in dirs:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)


def safe_delete_directory(
    dir_path: PathLike, max_retries: int = 5, retry_delay: float = 2.0
) -> bool:
    path = Path(dir_path)
    if not path.exists():
        return True

    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            return True
        except Exception:
            time.sleep(retry_delay)
    return False


def get_file_hash(file_path: PathLike) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)
    text = re.sub(r"(?<=\d)。(?=\d)", ".", text)
    text = re.sub(r"(?<!\d)。(?!\d)", ".", text)
    text = text.replace("，", ",").replace("；", ";")
    return text


def split_text_by_sentence(text: str, max_len: int = 200, min_len: int = 20) -> List[str]:
    if not text:
        return []
    sentence_seps = r"(?<!\d)([。！？；.!?;])(?![0-9.])"
    parts = [p.strip() for p in re.split(sentence_seps, text) if p.strip()]

    sentences: List[str] = []
    i = 0
    while i < len(parts):
        content = parts[i]
        sep = parts[i + 1] if (i + 1 < len(parts) and parts[i + 1] in "。！？；.!?;") else "."
        sentences.append(f"{content}{sep}")
        i += 2

    merged: List[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= max_len and current:
            current = current[:-1] + sent
        else:
            if current:
                merged.append(current)
            current = sent
    if current:
        merged.append(current)

    return [s for s in merged if len(s) >= min_len]


def load_json_file(file_path: PathLike, default: Any = None) -> Any:
    path = Path(file_path)
    if not path.exists():
        return default if default is not None else {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def save_json_file(data: Any, file_path: PathLike) -> bool:
    path = Path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def calculate_risk_level(score: float) -> str:
    if score < 30:
        return "低风险"
    if score < 60:
        return "中风险"
    if score < 90:
        return "高风险"
    return "极高风险"


def setup_logger(
    name: str, log_file: Optional[str] = None, level: int = logging.INFO
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
