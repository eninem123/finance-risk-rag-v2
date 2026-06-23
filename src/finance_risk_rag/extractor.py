"""
Finance-Risk-RAG 实体提取模块
============================
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

from .config import get_config
from .exceptions import ExtractionError
from .models import Entity, ExtractionResult
from .utils import calculate_risk_level, clean_text, load_json_file

logger = logging.getLogger(__name__)


class RuleBasedExtractor:
    """基于规则的实体提取器"""

    def __init__(self, config=None, rules_path: Optional[Path] = None):
        self.config = config or get_config()
        self.rules = {}
        if rules_path:
            self.load_rules(rules_path)
        elif self.config.risk_entities_path.exists():
            self.load_rules(self.config.risk_entities_path)

    def load_rules(self, rules_path: Path):
        try:
            self.rules = load_json_file(rules_path)
            logger.info(f"Loaded {len(self.rules)} rule categories.")
        except Exception as e:
            raise ExtractionError(f"Failed to load rules: {e}")

    def extract(self, text: str) -> List[Entity]:
        if not text or not self.rules:
            return []

        entities: List[Entity] = []
        seen: Set[Tuple[str, str, int]] = set()

        for entity_type, config in self.rules.items():
            keywords = config.get("keywords", [])
            base_risk_score = config.get("risk_score", 10)

            for keyword in keywords:
                # Use regex for better matching, but avoid \b for CJK
                # Basic implementation: find all occurrences
                for match in re.finditer(re.escape(keyword), text, re.IGNORECASE):
                    start = match.start()
                    end = match.end()
                    key = (entity_type, keyword, start)
                    if key in seen:
                        continue
                    seen.add(key)

                    context_start = max(0, start - 80)
                    context_end = min(len(text), end + 80)
                    context = text[context_start:context_end].replace("\n", " ").strip()

                    entities.append(
                        Entity(
                            type=entity_type,
                            text=keyword,
                            risk_score=base_risk_score,
                            confidence=1.0,
                            context=context,
                            source="rule",
                            start_char=start,
                            end_char=end,
                        )
                    )
        return entities


class BERTExtractor:
    """基于 BERT 的实体提取器"""

    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.tokenizer = None
        self.device = None
        if model_path and model_path.exists():
            self.load_model(model_path)

    def load_model(self, model_path: Path):
        try:
            import torch
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model = AutoModelForTokenClassification.from_pretrained(str(model_path))
            self.device = 0 if torch.cuda.is_available() else -1
            self.nlp = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                aggregation_strategy="simple",
            )
            logger.info(f"BERT model loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load BERT model: {e}")
            self.model = None

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def _chunk_text(
        self, text: str, max_chars: int = 1000, overlap: int = 200
    ) -> List[Tuple[str, int]]:
        """将文本切分为带重叠的字符块，返回 (块文本, 起始偏移量) 列表"""
        chunks = []
        if not text:
            return chunks

        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + max_chars, text_len)
            chunks.append((text[start:end], start))
            if end == text_len:
                break
            start += max_chars - overlap
        return chunks

    def extract(self, text: str) -> List[Entity]:
        if not self.is_available or not text:
            return []

        try:
            # 对于长文本，使用滑动窗口切分
            chunks = self._chunk_text(text)
            all_entities = []
            seen_keys = set()

            for chunk_text, offset in chunks:
                results = self.nlp(chunk_text)
                for res in results:
                    # 转换偏移量回全局坐标
                    global_start = res["start"] + offset
                    global_end = res["end"] + offset

                    # 使用 (类型, 文本, 开始位置) 作为唯一键进行去重
                    key = (res["entity_group"], res["word"], global_start)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    all_entities.append(
                        Entity(
                            type=res["entity_group"],
                            text=res["word"],
                            risk_score=20,
                            confidence=float(res["score"]),
                            context=text[
                                max(0, global_start - 40) : min(len(text), global_end + 40)
                            ],
                            source="bert",
                            start_char=global_start,
                            end_char=global_end,
                        )
                    )
            return all_entities
        except Exception as e:
            logger.error(f"BERT extraction failed: {e}")
            return []


class EntityExtractionPipeline:
    """实体提取管道"""

    def __init__(self, config=None, rule_extractor=None, bert_extractor=None):
        self.config = config or get_config()
        self.rule_extractor = rule_extractor or RuleBasedExtractor(config=self.config)
        self.bert_extractor = bert_extractor or BERTExtractor(self.config.bert_local_path)

    def process(self, text_or_path: Union[str, Path]) -> ExtractionResult:
        if isinstance(text_or_path, Path):
            text = text_or_path.read_text(encoding="utf-8")
        else:
            text = text_or_path

        text = clean_text(text)

        rule_entities = self.rule_extractor.extract(text)
        bert_entities = self.bert_extractor.extract(text)

        # Advanced merging logic: Score-based arbitration and overlap resolution
        entities_list = self._merge_and_arbitrate(rule_entities, bert_entities)

        total_risk = sum(e.risk_score for e in entities_list)
        risk_level = calculate_risk_level(total_risk)

        return ExtractionResult(
            entities=entities_list, total_risk_score=total_risk, risk_level=risk_level
        )

    def _merge_and_arbitrate(
        self, rule_entities: List[Entity], bert_entities: List[Entity]
    ) -> List[Entity]:
        """
        合并规则引擎和 BERT 的结果，处理重叠。
        仲裁逻辑：
        1. 优先考虑高置信度 (score > 0.85) 的 BERT 实体
        2. 其次按风险分数排序
        3. 最后按置信度和长度排序
        """
        all_entities = rule_entities + bert_entities
        if not all_entities:
            return []

        # 排序权重
        def arbitration_score(e: Entity):
            # 优先高置信度 BERT 实体
            bert_bonus = 1000 if (e.source == "bert" and e.confidence > 0.85) else 0
            return (bert_bonus, e.risk_score, e.confidence, len(e.text))

        all_entities.sort(key=arbitration_score, reverse=True)

        final_entities: List[Entity] = []

        for current in all_entities:
            is_redundant = False
            for existing in final_entities:
                # 精确字符重叠检测
                overlap_len = min(current.end_char, existing.end_char) - max(
                    current.start_char, existing.start_char
                )
                if overlap_len > 0:
                    # 如果重叠，由于已排序，当前实体被视为冗余
                    is_redundant = True
                    break

            if not is_redundant:
                final_entities.append(current)

        return final_entities
