"""
Finance-Risk-RAG 实体提取模块
============================

从财务文档中提取风险实体，支持规则提取和BERT模型提取。

功能:
    - 规则实体提取（基于关键词匹配）
    - BERT模型实体识别
    - 实体融合与去重
    - RAG向量库构建
    - 风险问答系统

作者: Finance-Risk-RAG Team
版本: 2.0.0
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

from config import get_config
from utils import (
    clean_text,
    calculate_risk_level,
    ensure_dirs,
    load_json_file,
    save_json_file,
    setup_logger,
    safe_delete_directory,
)


# ==================== 数据类定义 ====================

@dataclass
class Entity:
    """风险实体数据类"""
    type: str
    text: str
    risk_score: int
    confidence: float
    context: str = ""
    source: str = "rule"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": self.type,
            "text": self.text,
            "risk_score": self.risk_score,
            "confidence": round(self.confidence, 4),
            "context": self.context,
            "source": self.source,
            **self.metadata
        }
    
    @property
    def key(self) -> Tuple[str, str]:
        """实体唯一键（用于去重）"""
        return (self.text, self.type)


@dataclass
class ExtractionResult:
    """提取结果数据类"""
    entities: List[Entity]
    total_risk_score: int
    risk_level: str
    extraction_time: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "extracted_at": self.extraction_time,
            "total_entities": len(self.entities),
            "total_risk_score": self.total_risk_score,
            "risk_level": self.risk_level,
            "entities": [e.to_dict() for e in self.entities],
            **self.metadata
        }


# ==================== 异常定义 ====================

class ExtractionError(Exception):
    """实体提取异常"""
    pass


class RuleLoadError(ExtractionError):
    """规则加载异常"""
    pass


# ==================== 规则实体提取器 ====================

class RuleBasedExtractor:
    """基于规则的实体提取器"""
    
    # 数值提取模式
    NUM_PATTERNS = {
        "liquidity_risk": r'(现金储备|现金及现金等价物|cash.*reserve).*?(\d+[,\d]*\.?\d*)\s*(亿|亿元|百万|million|billion)',
        "credit_rating": r'(评级|rating).*?(AAA|AA\+|AA|AA-|A\+|A|A-|BBB\+|BBB|BBB-)',
        "contingent_liability": r'(诉讼|pending litigation).*?(\d+[,\d]*\.?\d*)\s*(亿|万元|USD)',
        "related_transaction": r'(关联交易金额|related party).*?(\d+[,\d]*\.?\d*)\s*(亿|万元|HKD|USD)'
    }
    
    def __init__(self, rules_path: Optional[Path] = None) -> None:
        """
        初始化规则提取器
        
        Args:
            rules_path: 规则文件路径
        """
        self._rules: Dict[str, Any] = {}
        self._logger = logging.getLogger(__name__)
        
        if rules_path:
            self.load_rules(rules_path)
    
    def load_rules(self, rules_path: Path) -> None:
        """
        加载实体规则
        
        Args:
            rules_path: 规则文件路径
            
        Raises:
            RuleLoadError: 规则加载失败
        """
        try:
            self._rules = load_json_file(rules_path)
            self._logger.info(f"加载规则成功: {len(self._rules)} 类")
        except Exception as e:
            self._logger.error(f"规则加载失败: {e}")
            raise RuleLoadError(f"无法加载规则文件: {e}") from e
    
    def extract(self, text: str) -> List[Entity]:
        """
        从文本中提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            提取的实体列表
        """
        if not text or not self._rules:
            return []
        
        entities: List[Entity] = []
        seen: Set[Tuple[str, str, int]] = set()
        
        # 关键词匹配
        for entity_type, config in self._rules.items():
            keywords = config.get("keywords", [])
            base_risk_score = config.get("risk_score", 10)
            
            for keyword in keywords:
                pattern = rf'\b{re.escape(keyword)}\b'
                
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start = match.start()
                    key = (entity_type, keyword, start)
                    
                    if key in seen:
                        continue
                    seen.add(key)
                    
                    # 提取上下文
                    context_start = max(0, start - 80)
                    context_end = min(len(text), start + len(keyword) + 80)
                    context = text[context_start:context_end].replace("\n", " ").strip()
                    
                    entities.append(Entity(
                        type=entity_type,
                        text=keyword,
                        risk_score=base_risk_score,
                        confidence=1.0,
                        context=context,
                        source="rule"
                    ))
        
        return entities


# ==================== BERT 实体提取器 ====================

class BERTExtractor:
    """基于BERT的实体提取器"""
    
    def __init__(self, model_path: Optional[Path] = None) -> None:
        """
        初始化BERT提取器
        
        Args:
            model_path: 模型路径
        """
        self._model = None
        self._tokenizer = None
        self._device = None
        self._logger = logging.getLogger(__name__)
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: Path) -> bool:
        """
        加载BERT模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            是否加载成功
        """
        try:
            # 延迟导入以减少启动时间
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
            
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self._model = AutoModelForTokenClassification.from_pretrained(str(model_path))
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
            self._model.eval()
            
            self._logger.info(f"BERT模型加载成功: {model_path}")
            return True
        except Exception as e:
            self._logger.warning(f"BERT模型加载失败: {e}")
            return False
    
    @property
    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self._model is not None
    
    def extract(
        self,
        text: str,
        chunk_size: int = 400,
        overlap: int = 50
    ) -> List[Entity]:
        """
        使用BERT提取实体
        
        Args:
            text: 输入文本
            chunk_size: 分块大小
            overlap: 重叠大小
            
        Returns:
            提取的实体列表
        """
        if not self.is_available or not text:
            return []
        
        entities: List[Entity] = []
        chunks = self._chunk_text(text, chunk_size, overlap)
        
        for i, chunk in enumerate(chunks):
            try:
                chunk_entities = self._extract_from_chunk(chunk, f"chunk_{i}")
                entities.extend(chunk_entities)
                time.sleep(0.1)  # 避免过载
            except Exception as e:
                self._logger.warning(f"分块 {i} 提取失败: {e}")
        
        return entities
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """文本分块"""
        words = text.split()
        chunks: List[str] = []
        i = 0
        
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap
            
            if i >= len(words) and len(chunks) > 1:
                break
        
        return chunks
    
    def _extract_from_chunk(
        self,
        chunk: str,
        chunk_id: str
    ) -> List[Entity]:
        """从单个分块提取实体"""
        # 这里应该实现实际的BERT NER逻辑
        # 简化版本返回空列表
        return []


# ==================== 实体融合器 ====================

class EntityMerger:
    """实体融合器"""
    
    def merge(
        self,
        rule_entities: List[Entity],
        bert_entities: List[Entity]
    ) -> List[Entity]:
        """
        融合规则提取和BERT提取的实体
        
        Args:
            rule_entities: 规则提取的实体
            bert_entities: BERT提取的实体
            
        Returns:
            融合后的实体列表
        """
        merged: Dict[Tuple[str, str], Entity] = {}
        
        for entity in rule_entities + bert_entities:
            key = entity.key
            
            if key not in merged:
                merged[key] = entity
            else:
                # 合并信息，取更高的置信度和风险分数
                existing = merged[key]
                existing.confidence = max(existing.confidence, entity.confidence)
                existing.risk_score = max(existing.risk_score, entity.risk_score)
        
        return list(merged.values())


# ==================== RAG 问答系统 ====================

class RAGQAService:
    """RAG问答服务"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "moonshot-v1-8k"
    ) -> None:
        """
        初始化问答服务
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model_name: 模型名称
        """
        config = get_config()
        self._client = OpenAI(
            api_key=api_key or config.llm_api_key,
            base_url=base_url or config.llm_base_url
        )
        self._model_name = model_name
        self._logger = logging.getLogger(__name__)
    
    def query(
        self,
        question: str,
        context_entities: List[Entity],
        max_tokens: int = 500
    ) -> str:
        """
        执行问答
        
        Args:
            question: 用户问题
            context_entities: 上下文实体
            max_tokens: 最大token数
            
        Returns:
            回答
        """
        if not context_entities:
            return "未发现相关风险实体。"
        
        # 构建上下文
        context = "\n".join([
            f"【{e.type}】{e.text} (风险分: {e.risk_score}, 置信: {e.confidence:.2f})"
            for e in context_entities[:5]
        ])
        
        prompt = f"""
你是一个银行风控专家。基于以下检索到的风险实体，简洁、专业地回答问题。

【检索到的风险实体】：
{context}

【用户问题】：
{question}

【要求】：
- 回答简洁、专业、数据化
- 若无数据，说"未发现相关风险"
- 输出纯文本，无 JSON

回答：
"""
        
        try:
            time.sleep(1)  # 避免API限流
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self._logger.error(f"问答失败: {e}")
            return f"问答失败：{e}"


# ==================== 实体提取管道 ====================

class EntityExtractionPipeline:
    """实体提取管道"""
    
    def __init__(self, config: Optional[Any] = None) -> None:
        """
        初始化管道
        
        Args:
            config: 配置对象
        """
        self._config = config or get_config()
        self._logger = setup_logger("entity_extraction", "logs/extract_entities.log")
        
        # 初始化组件
        self._rule_extractor = RuleBasedExtractor()
        self._bert_extractor = BERTExtractor()
        self._merger = EntityMerger()
        self._qa_service: Optional[RAGQAService] = None
    
    def initialize(self) -> None:
        """初始化管道组件"""
        # 加载规则
        rules_path = self._config.risk_entities_path
        if rules_path.exists():
            self._rule_extractor.load_rules(rules_path)
        
        # 加载BERT模型
        if self._config.bert_local_path:
            self._bert_extractor.load_model(self._config.bert_local_path)
        
        # 初始化问答服务
        if self._config.llm_api_key:
            self._qa_service = RAGQAService()
        
        self._logger.info("实体提取管道初始化完成")
    
    def process(self, text_path: Path) -> ExtractionResult:
        """
        处理文本文件
        
        Args:
            text_path: 文本文件路径
            
        Returns:
            提取结果
        """
        self._logger.info(f"开始处理: {text_path}")
        
        # 读取文本
        if not text_path.exists():
            raise ExtractionError(f"文本文件不存在: {text_path}")
        
        text = text_path.read_text(encoding="utf-8")
        text = clean_text(text)
        
        self._logger.info(f"文本长度: {len(text)} 字符")
        
        # 规则提取
        self._logger.info("执行规则提取...")
        rule_entities = self._rule_extractor.extract(text)
        self._logger.info(f"规则提取实体数: {len(rule_entities)}")
        
        # BERT提取
        bert_entities: List[Entity] = []
        if self._bert_extractor.is_available:
            self._logger.info("执行BERT提取...")
            bert_entities = self._bert_extractor.extract(text)
            self._logger.info(f"BERT提取实体数: {len(bert_entities)}")
        
        # 融合
        final_entities = self._merger.merge(rule_entities, bert_entities)
        self._logger.info(f"融合后实体数: {len(final_entities)}")
        
        # 计算风险
        total_risk = sum(e.risk_score for e in final_entities)
        risk_level = calculate_risk_level(total_risk)
        
        return ExtractionResult(
            entities=final_entities,
            total_risk_score=total_risk,
            risk_level=risk_level,
            metadata={
                "rule_entities": len(rule_entities),
                "bert_entities": len(bert_entities)
            }
        )
    
    def save_result(
        self,
        result: ExtractionResult,
        output_path: Path
    ) -> None:
        """保存提取结果"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json_file(result.to_dict(), output_path)
        self._logger.info(f"结果已保存: {output_path}")
    
    def interactive_qa(self, entities: List[Entity]) -> None:
        """交互式问答"""
        if not self._qa_service:
            print("问答服务未初始化，请设置API密钥")
            return
        
        print("\nRAG 风控问答系统已就绪！输入 'exit' 退出。")
        
        while True:
            try:
                question = input("\n问：").strip()
                
                if question.lower() in ["exit", "quit", "退出"]:
                    print("再见！")
                    break
                
                if not question:
                    continue
                
                answer = self._qa_service.query(question, entities)
                print(f"答：{answer}")
                
            except KeyboardInterrupt:
                print("\n再见！")
                break


# ==================== 命令行入口 ====================

def main() -> None:
    """命令行入口函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Finance-Risk-RAG 实体提取")
    parser.add_argument("--input", type=str, default="docs/all_extracted.txt", help="输入文本文件")
    parser.add_argument("--output", type=str, default="docs/entities_extracted.json", help="输出JSON文件")
    parser.add_argument("--no-qa", action="store_true", help="禁用交互式问答")
    
    args = parser.parse_args()
    
    # 初始化管道
    pipeline = EntityExtractionPipeline()
    pipeline.initialize()
    
    # 处理文本
    result = pipeline.process(Path(args.input))
    
    # 保存结果
    pipeline.save_result(result, Path(args.output))
    
    # 打印摘要
    print(f"\n实体提取完成！")
    print(f"  实体数: {len(result.entities)}")
    print(f"  总风险: {result.total_risk_score}/100 ({result.risk_level})")
    print(f"  保存: {args.output}")
    
    # 打印Top 5高风险实体
    print("\nTop 5 高风险实体：")
    top5 = sorted(result.entities, key=lambda x: x.risk_score, reverse=True)[:5]
    for e in top5:
        print(f"  {e.type:20} | {e.text:30} | 分数: {e.risk_score:2} | 置信: {e.confidence:.2f}")
    
    # 交互式问答
    if not args.no_qa:
        pipeline.interactive_qa(result.entities)


if __name__ == "__main__":
    main()
