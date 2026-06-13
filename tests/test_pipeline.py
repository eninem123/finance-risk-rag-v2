"""
提取流水线单元测试
"""

import unittest
from unittest.mock import MagicMock

from src.finance_risk_rag.extractor import EntityExtractionPipeline
from src.finance_risk_rag.models import Entity


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock()
        self.mock_rule = MagicMock()
        self.mock_bert = MagicMock()
        self.pipeline = EntityExtractionPipeline(
            config=self.mock_config,
            rule_extractor=self.mock_rule,
            bert_extractor=self.mock_bert,
        )

    def test_merge_and_arbitrate_overlap(self):
        # 准备两个重叠的实体
        e1 = Entity(
            type="RISK",
            text="财务风险",
            risk_score=30,
            confidence=1.0,
            metadata={"start_char": 0, "end_char": 4},
        )
        e2 = Entity(
            type="RISK",
            text="风险",
            risk_score=10,
            confidence=0.9,
            metadata={"start_char": 2, "end_char": 4},
        )

        merged = self.pipeline._merge_and_arbitrate([e1], [e2])

        # 应该只保留分数高且覆盖范围大的 e1
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "财务风险")

    def test_process_logic(self):
        self.mock_rule.extract.return_value = []
        self.mock_bert.extract.return_value = []

        result = self.pipeline.process("文本样本")

        self.assertEqual(result.total_risk_score, 0)
        self.assertEqual(result.risk_level, "低风险")


if __name__ == "__main__":
    unittest.main()
