"""
Utility functions for the Finance-Risk-RAG system.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import jieba

PathLike = Union[str, Path]


def setup_logger(
    name: str,
    log_file: Optional[PathLike] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Configure and return a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_path), encoding='utf-8')
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing punctuation."""
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Normalizing common Chinese punctuation
    text = text.replace('，', ',').replace('；', ';').replace('。', '.')

    return text


def split_text_by_sentence(text: str, max_len: int = 200) -> List[str]:
    """Split text into sentences, roughly respecting a max length."""
    if not text:
        return []

    # Split by common sentence delimiters (avoiding splitting numbers with dots)
    sentence_seps = r'(?<!\d)([.!?;])(?![0-9])'
    parts = re.split(sentence_seps, text)

    sentences = []
    current = ""

    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i].strip() + parts[i+1]
        if len(current) + len(sentence) <= max_len:
            current += " " + sentence if current else sentence
        else:
            if current:
                sentences.append(current.strip())
            current = sentence

    if current:
        sentences.append(current.strip())

    return [s for s in sentences if s]


def get_file_hash(file_path: PathLike) -> str:
    """Calculate the MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_json_file(file_path: PathLike, default: Any = None) -> Any:
    """Safely load a JSON file."""
    path = Path(file_path)
    if not path.exists():
        return default if default is not None else {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def save_json_file(data: Any, file_path: PathLike) -> bool:
    """Safely save data to a JSON file."""
    path = Path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def calculate_risk_level(score: int) -> str:
    """Determine risk level based on total score."""
    if score < 30:
        return "Low Risk"
    elif score < 60:
        return "Medium Risk"
    elif score < 90:
        return "High Risk"
    else:
        return "Extreme Risk"


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract keywords using jieba."""
    if not text:
        return []
    import jieba.analyse
    return jieba.analyse.extract_tags(text, topK=top_n)
