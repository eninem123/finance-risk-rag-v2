"""
Finance-Risk-RAG Utility Module
==============================

General utility functions for text processing, file operations, and logging.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, List, Optional, Union


def setup_logger(
    name: str, log_file: Optional[Union[str, Path]] = None, level: int = logging.INFO
) -> logging.Logger:
    """Configure and return a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def get_file_hash(file_path: Union[str, Path]) -> str:
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Normalize CJK punctuation for regex compatibility
    text = (
        text.replace("。", ".")
        .replace("，", ",")
        .replace("；", ";")
        .replace("！", "!")
        .replace("？", "?")
    )
    return text


def split_text_by_sentence(text: str, max_len: int = 500) -> List[str]:
    """Split text into sentences, respecting max length."""
    # Simple split by punctuation
    # Use regex to keep delimiters
    parts = re.split(r"([.!?;!！?？])", text)

    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentences.append(parts[i] + parts[i + 1])
    if len(parts) % 2 == 1 and parts[-1]:
        sentences.append(parts[-1])

    result = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) <= max_len:
            current += " " + sentence if current else sentence
        else:
            if current:
                result.append(current.strip())
            current = sentence

    if current:
        result.append(current.strip())

    return [s for s in result if s]


def load_json(file_path: Union[str, Path], default: Any = None) -> Any:
    """Safely load a JSON file."""
    path = Path(file_path)
    if not path.exists():
        return default if default is not None else {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def save_json(data: Any, file_path: Union[str, Path]) -> bool:
    """Safely save data to a JSON file."""
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def calculate_risk_level(score: float) -> str:
    """Map a risk score to a qualitative level."""
    if score < 30:
        return "Low Risk"
    if score < 60:
        return "Medium Risk"
    if score < 90:
        return "High Risk"
    return "Extreme Risk"
