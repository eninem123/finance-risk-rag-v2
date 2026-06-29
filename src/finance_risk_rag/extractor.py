"""
Finance-Risk-RAG 实体提取模块
============================
"""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

from .config import get_config
from .exceptions import ExtractionError
from .models import Entity, ExtractionResult
from .utils import calculate_risk_level, clean_text, load_json_file

logger = logging.getLogger(__name__)


class ScoringStrategy(ABC):
    """评分策略基类"""

    @abstractmethod
    def calculate(self, entity: Entity, context_text: str) -> Entity:
        pass


class FinanceRiskScoringStrategy(ScoringStrategy):
    """金融级多维度评分策略"""

    def __init__(self):
        # 核心金融风险关键词及其权重
        self.risk_boosters = {
            "逾期": 1.5,
            "亏损": 1.4,
            "违约": 1.6,
            "资不抵债": 1.8,
            "诉讼": 1.3,
            "负债": 1.2,
            "风险": 1.2,
            "破产": 1.9,
        }

        # 风险分类映射
        self.category_map = {
            "RISK": "一般风险",
            "MONEY": "财务风险",
            "ORG": "机构风险",
            "PER": "个人风险",
            "CREDIT": "信用风险",
            "LEGAL": "合规/法律风险",
        }

    def calculate(self, entity: Entity, context_text: str) -> Entity:
        # 1. 基础评分调整：结合置信度
        score = entity.risk_score * (0.5 + 0.5 * entity.confidence)

        # 2. 关键词加权：检测上下文中的风险触发词
        impact_boost = 1.0
        # 优先使用 entity.context，如果为空则使用 context_text (但在本系统中 extractor 应该已经填充了 context)
        search_text = entity.context or context_text

        for word, boost in self.risk_boosters.items():
            if word in entity.text or word in search_text:
                score *= boost
                impact_boost = max(impact_boost, boost * 2)

        # 3. 映射风险分类
        entity.risk_category = self.category_map.get(entity.type.upper(), "其他风险")

        # 4. 设置最终评分和影响度
        entity.risk_score = int(min(100, score))
        entity.impact_score = min(5.0, impact_boost)

        return entity


class BaseExtractor(ABC):
    """提取器基类"""

    @abstractmethod
    def extract(self, text: str) -> List[Entity]:
        pass


class RuleBasedExtractor(BaseExtractor):
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


class BERTExtractor(BaseExtractor):
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

    def extract(self, text: str) -> List[Entity]:
        if not self.is_available or not text:
            return []

        try:
            results = self.nlp(text)
            entities = []
            for res in results:
                start = res["start"]
                end = res["end"]
                context_start = max(0, start - 40)
                context_end = min(len(text), end + 40)
                context = text[context_start:context_end].replace("\n", " ").strip()

                entities.append(
                    Entity(
                        type=res["entity_group"],
                        text=res["word"],
                        risk_score=20,  # Default risk score for BERT entities
                        confidence=float(res["score"]),
                        context=context,
                        source="bert",
                        start_char=start,
                        end_char=end,
                    )
                )
            return entities
        except Exception as e:
            logger.error(f"BERT extraction failed: {e}")
            return []


class EntityExtractionPipeline:
    """实体提取管道"""

    def __init__(
        self,
        config=None,
        rule_extractor=None,
        bert_extractor=None,
        scoring_strategy=None,
    ):
        self.config = config or get_config()
        self.rule_extractor = rule_extractor or RuleBasedExtractor(config=self.config)
        self.bert_extractor = bert_extractor or BERTExtractor(self.config.bert_local_path)
        self.scoring_strategy = scoring_strategy or FinanceRiskScoringStrategy()

    def process(self, text_or_path: Union[str, Path]) -> ExtractionResult:
        if isinstance(text_or_path, Path):
            text = text_or_path.read_text(encoding="utf-8")
        else:
            text = text_or_path

        text = clean_text(text)

        rule_entities = self.rule_extractor.extract(text)
        bert_entities = self.bert_extractor.extract(text)

        # 1. 合并并仲裁重叠
        merged_entities = self._merge_and_arbitrate(rule_entities, bert_entities)

        # 2. 应用高级评分策略
        final_entities = [
            self.scoring_strategy.calculate(e, text) for e in merged_entities
        ]

        total_risk = sum(e.risk_score for e in final_entities)
        risk_level = calculate_risk_level(total_risk)

        return ExtractionResult(
            entities=final_entities, total_risk_score=total_risk, risk_level=risk_level
        )

    def _merge_and_arbitrate(
        self, rule_entities: List[Entity], bert_entities: List[Entity]
    ) -> List[Entity]:
        """
        合并规则引擎和 BERT 的结果，处理重叠。
        优先考虑高分和高置信度的实体。
        """
        all_entities = rule_entities + bert_entities
        if not all_entities:
            return []

        # 按得分和置信度排序
        all_entities.sort(key=lambda x: (x.risk_score, x.confidence), reverse=True)

        final_entities: List[Entity] = []

        for current in all_entities:
            is_redundant = False
            for existing in final_entities:
                overlap = max(current.start_char, existing.start_char) < min(
                    current.end_char, existing.end_char
                )
                if overlap:
                    is_redundant = True
                    break

            if not is_redundant:
                final_entities.append(current)

        return final_entities
