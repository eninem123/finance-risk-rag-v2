import unittest

from src.finance_risk_rag.extractor import EntityExtractionPipeline
from src.finance_risk_rag.models import Entity


class TestArbitration(unittest.TestCase):
    def setUp(self):
        self.pipeline = EntityExtractionPipeline()

    def test_merge_and_arbitrate_priority(self):
        # Case: Overlapping entities, BERT has high confidence (> 0.85)
        # Rule entity: "bad debt", risk 30, confidence 1.0, span (0, 8)
        # BERT entity: "bad debt", risk 20, confidence 0.9, span (0, 8)

        rule_e = Entity(
            type="RISK",
            text="bad debt",
            risk_score=30,
            confidence=1.0,
            source="rule",
            start_char=0,
            end_char=8,
        )
        bert_e = Entity(
            type="MONEY",
            text="bad debt",
            risk_score=20,
            confidence=0.9,
            source="bert",
            start_char=0,
            end_char=8,
        )

        # High confidence BERT entities (> 0.85) should take precedence.
        results = self.pipeline._merge_and_arbitrate([rule_e], [bert_e])

        self.assertEqual(results[0].source, "bert")
        self.assertEqual(len(results), 1)

    def test_no_overlap(self):
        e1 = Entity(
            type="RISK", text="debt", risk_score=30, confidence=1.0, start_char=0, end_char=4
        )
        e2 = Entity(
            type="RISK", text="risk", risk_score=30, confidence=1.0, start_char=10, end_char=14
        )

        results = self.pipeline._merge_and_arbitrate([e1], [e2])
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
