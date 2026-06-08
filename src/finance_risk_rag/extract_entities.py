"""
Finance-Risk-RAG 实体提取模块
"""

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

from finance_risk_rag.config import get_config
from finance_risk_rag.models import Entity, ExtractionResult
from finance_risk_rag.utils import (
    clean_text,
    calculate_risk_level,
    load_json_file,
    save_json_file,
    setup_logger,
)


class RuleBasedExtractor:
    """基于规则的实体提取器"""

    def __init__(self, rules_path: Optional[Path] = None):
        self.rules = load_json_file(rules_path) if rules_path else {}
        self.logger = setup_logger("rule_extractor")

    def extract(self, text: str) -> List[Entity]:
        if not text or not self.rules:
            return []

        entities = []
        for entity_type, config in self.rules.items():
            keywords = config.get("keywords", [])
            score = config.get("risk_score", 10)

            for keyword in keywords:
                # 注意：对中文不使用 \b
                pattern = re.escape(keyword)
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start = match.start()
                    context = text[max(0, start-40):min(len(text), start+len(keyword)+40)]

                    entities.append(Entity(
                        type=entity_type,
                        text=keyword,
                        risk_score=score,
                        confidence=1.0,
                        context=context.strip(),
                        source="rule"
                    ))
        return entities


class BERTExtractor:
    """基于 BERT 的实体提取器"""

    def __init__(self, model_path: Optional[Path] = None):
        self.pipeline = None
        self.logger = setup_logger("bert_extractor")
        if model_path and model_path.exists():
            self.load_model(model_path)

    def load_model(self, model_path: Path):
        try:
            from transformers import pipeline
            self.pipeline = pipeline(
                "token-classification",
                model=str(model_path),
                tokenizer=str(model_path),
                aggregation_strategy="simple"
            )
            self.logger.info(f"BERT 模型加载成功: {model_path}")
        except Exception as e:
            self.logger.error(f"BERT 模型加载失败: {e}")

    @property
    def is_available(self) -> bool:
        return self.pipeline is not None

    def _chunk_text(self, text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
        """将文本分块以适应 BERT 输入限制"""
        # 简单按字符分块（对于 BERT 来说通常是 token，但这里近似处理）
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_length
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start += max_length - overlap
        return chunks

    def extract(self, text: str) -> List[Entity]:
        if not self.is_available or not text:
            return []

        # 风险分数映射
        score_map = {"RISK": 30, "MONEY": 25, "ORG": 15, "PER": 5}
        entities = []

        # 分块处理长文本
        chunks = self._chunk_text(text)
        for chunk in chunks:
            try:
                results = self.pipeline(chunk)
                for res in results:
                    ent_type = res.get("entity_group", "UNKNOWN")
                    score = score_map.get(ent_type, 10)

                    entities.append(Entity(
                        type=ent_type,
                        text=res.get("word", ""),
                        risk_score=score,
                        confidence=float(res.get("score", 0.0)),
                        context="",
                        source="bert"
                    ))
            except Exception as e:
                self.logger.error(f"BERT 分块提取失败: {e}")

        return entities


class EntityExtractionPipeline:
    """实体提取管道"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = setup_logger("extraction_pipeline", str(self.config.log_dir / "extraction.log"))

        self.rule_extractor = RuleBasedExtractor(self.config.risk_entities_path)
        self.bert_extractor = BERTExtractor(self.config.bert_local_path)
        self.client = None
        if self.config.llm_api_key:
            self.client = OpenAI(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url
            )

    def initialize(self):
        pass # 初始化逻辑已在构造函数或通过 config 处理

    def process(self, text_path: Path) -> ExtractionResult:
        if not text_path.exists():
            return ExtractionResult([], 0, "未知")

        text = text_path.read_text(encoding="utf-8")
        text = clean_text(text)

        # 规则提取
        rule_entities = self.rule_extractor.extract(text)

        # BERT 提取
        bert_entities = []
        if self.bert_extractor.is_available:
            bert_entities = self.bert_extractor.extract(text)

        # 融合与去重
        merged = {}
        for e in rule_entities + bert_entities:
            key = (e.text, e.type)
            if key not in merged or e.confidence > merged[key].confidence:
                merged[key] = e

        final_entities = list(merged.values())
        total_score = sum(e.risk_score for e in final_entities)
        risk_level = calculate_risk_level(total_score)

        return ExtractionResult(
            entities=final_entities,
            total_risk_score=total_score,
            risk_level=risk_level
        )

    def save_result(self, result: ExtractionResult, output_path: Path):
        save_json_file(result.to_dict(), output_path)

    def interactive_qa(self, entities: List[Entity]):
        if not self.client:
            print("LLM 未配置，无法进行问答")
            return

        print("\n--- 交互式风险问答 (输入 'exit' 退出) ---")
        context = "\n".join([f"- {e.text} ({e.type}, 风险分: {e.risk_score})" for e in entities[:10]])

        while True:
            q = input("\n问：").strip()
            if q.lower() in ["exit", "quit"]: break
            if not q: continue

            prompt = f"基于以下风险实体回答：\n{context}\n\n问题：{q}"
            try:
                response = self.client.chat.completions.create(
                    model=self.config.llm_model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                print(f"答：{response.choices[0].message.content}")
            except Exception as e:
                print(f"问答失败: {e}")
