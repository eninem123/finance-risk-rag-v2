import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from finance_risk_rag.config import get_config
from finance_risk_rag.exceptions import ExtractionError, RuleLoadError
from finance_risk_rag.models import Entity, ExtractionResult
from finance_risk_rag.utils import (
    calculate_risk_level,
    clean_text,
    load_json_file,
    save_json_file,
    setup_logger,
)

logger = logging.getLogger(__name__)


class RuleBasedExtractor:
    """基于规则的实体提取器"""

    def __init__(self, rules_path: Optional[Path] = None) -> None:
        self._rules: Dict[str, Any] = {}
        if rules_path:
            self.load_rules(rules_path)

    def load_rules(self, rules_path: Path) -> None:
        try:
            self._rules = load_json_file(rules_path)
            logger.info(f"加载规则成功: {len(self._rules)} 类")
        except Exception as e:
            logger.error(f"规则加载失败: {e}")
            raise RuleLoadError(f"无法加载规则文件: {e}") from e

    def extract(self, text: str) -> List[Entity]:
        if not text or not self._rules:
            return []

        entities: List[Entity] = []
        seen: Set[Tuple[str, str, int]] = set()

        for entity_type, config in self._rules.items():
            keywords = config.get("keywords", [])
            base_risk_score = config.get("risk_score", 10)

            for keyword in keywords:
                # 兼容 CJK 和 英文 边界
                pattern = rf"{re.escape(keyword)}"

                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start = match.start()
                    key = (entity_type, keyword, start)

                    if key in seen:
                        continue
                    seen.add(key)

                    context_start = max(0, start - 80)
                    context_end = min(len(text), start + len(keyword) + 80)
                    context = text[context_start:context_end].replace("\n", " ").strip()

                    entities.append(
                        Entity(
                            type=entity_type,
                            text=keyword,
                            risk_score=base_risk_score,
                            confidence=1.0,
                            context=context,
                            source="rule",
                        )
                    )
        return entities


class BERTExtractor:
    """基于BERT的实体提取器"""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._nlp: Any = None
        self._logger = logging.getLogger(__name__)
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: Path) -> bool:
        try:
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1
            self._nlp = pipeline(
                "ner",
                model=str(model_path),
                tokenizer=str(model_path),
                device=device,
                aggregation_strategy="simple",
            )  # type: ignore[call-overload]
            self._logger.info(f"BERT模型加载成功: {model_path}")
            return True
        except Exception as e:
            self._logger.warning(f"BERT模型加载失败: {e}")
            return False

    @property
    def is_available(self) -> bool:
        return self._nlp is not None

    def extract(self, text: str) -> List[Entity]:
        if not self._nlp or not text:
            return []

        # 映射实体类型到风险评分
        risk_score_map = {
            "RISK": 30,
            "MONEY": 25,
            "ORG": 15,
            "PER": 5,
        }

        # BERT 通常有 512 token 限制，这里简单分块处理
        chunk_size = 500
        entities = []

        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            try:
                results = self._nlp(chunk)
                for res in results:
                    entity_type = res["entity_group"]
                    entities.append(
                        Entity(
                            type=entity_type,
                            text=res["word"],
                            risk_score=risk_score_map.get(entity_type, 20),
                            confidence=float(res["score"]),
                            context=chunk,
                            source="bert",
                        )
                    )
            except Exception as e:
                self._logger.warning(f"BERT 分块提取失败: {e}")

        return entities


class EntityMerger:
    """实体融合器"""

    def merge(self, rule_entities: List[Entity], bert_entities: List[Entity]) -> List[Entity]:
        merged: Dict[Tuple[str, str], Entity] = {}
        for entity in rule_entities + bert_entities:
            key = entity.key
            if key not in merged:
                merged[key] = entity
            else:
                existing = merged[key]
                existing.confidence = max(existing.confidence, entity.confidence)
                existing.risk_score = max(existing.risk_score, entity.risk_score)
        return list(merged.values())


class EntityExtractionPipeline:
    """实体提取管道"""

    def __init__(self) -> None:
        self._config = get_config()
        self._logger = setup_logger(
            "entity_extraction", str(self._config.log_dir / "extract_entities.log")
        )
        self._rule_extractor = RuleBasedExtractor()
        self._bert_extractor = BERTExtractor()
        self._merger = EntityMerger()
        self.initialize()

    def initialize(self) -> None:
        rules_path = self._config.risk_entities_path
        if rules_path.exists():
            self._rule_extractor.load_rules(rules_path)

        if self._config.bert_local_path:
            self._bert_extractor.load_model(self._config.bert_local_path)

        self._logger.info("实体提取管道初始化完成")

    def process(self, text_path: Path) -> ExtractionResult:
        if not text_path.exists():
            raise ExtractionError(f"文本文件不存在: {text_path}")

        text = text_path.read_text(encoding="utf-8")
        text = clean_text(text)

        rule_entities = self._rule_extractor.extract(text)
        bert_entities = self._bert_extractor.extract(text)
        final_entities = self._merger.merge(rule_entities, bert_entities)

        total_risk = sum(e.risk_score for e in final_entities)
        risk_level = calculate_risk_level(total_risk)

        return ExtractionResult(
            entities=final_entities,
            total_risk_score=total_risk,
            risk_level=risk_level,
            metadata={"rule_count": len(rule_entities), "bert_count": len(bert_entities)},
        )

    def save_result(self, result: ExtractionResult, output_path: Path) -> None:
        save_json_file(result.to_dict(), output_path)
