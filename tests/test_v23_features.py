import unittest
from pathlib import Path
from src.finance_risk_rag.extractor import (
    DefaultScoringStrategy,
    RuleBasedExtractor,
    BERTExtractor,
    ScoringStrategy
)
from src.finance_risk_rag.models import Entity

class CustomScoringStrategy(ScoringStrategy):
    def calculate_score(self, entity_type: str, confidence: float, base_score: int) -> int:
        if entity_type == "HIGH_RISK":
            return base_score * 2
        return base_score

class TestV23Features(unittest.TestCase):
    def test_default_scoring_strategy(self):
        strategy = DefaultScoringStrategy()
        # base 20, conf 0.8 -> 16
        self.assertEqual(strategy.calculate_score("test", 0.8, 20), 16)
        # base 20, conf 0.85 -> 17
        self.assertEqual(strategy.calculate_score("test", 0.85, 20), 17)
        # base 10, conf 1.0 -> 10
        self.assertEqual(strategy.calculate_score("test", 1.0, 10), 10)

    def test_custom_scoring_strategy(self):
        strategy = CustomScoringStrategy()
        self.assertEqual(strategy.calculate_score("HIGH_RISK", 1.0, 50), 100)
        self.assertEqual(strategy.calculate_score("NORMAL", 1.0, 50), 50)

    def test_bert_chunking_logic(self):
        # We can test the private _chunk_text method
        extractor = BERTExtractor()
        text = "A" * 1000
        # max_length=510, overlap=50
        # Chunk 1: 0-510
        # Chunk 2: (510-50)=460 to 460+510=970
        # Chunk 3: (970-50)=920 to 1000
        chunks = extractor._chunk_text(text, max_length=500, overlap=100)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], ("A" * 500, 0))
        self.assertEqual(chunks[1], ("A" * 500, 400))
        self.assertEqual(chunks[2], ("A" * 200, 800))

if __name__ == "__main__":
    unittest.main()
