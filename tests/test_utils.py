import unittest

from finance_risk_rag.utils import (
    calculate_risk_level,
    clean_text,
    split_text_by_sentence,
)


class TestUtils(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text("Hello   World"), "Hello World")
        self.assertEqual(clean_text("3。5"), "3.5")
        self.assertEqual(clean_text("项目，进展；"), "项目,进展;")

    def test_split_text_by_sentence(self):
        text = "这是第一句。这是第二句！那是第三句吗？"
        sentences = split_text_by_sentence(text)
        self.assertTrue(len(sentences) >= 1)

    def test_calculate_risk_level(self):
        self.assertEqual(calculate_risk_level(10), "低风险")
        self.assertEqual(calculate_risk_level(45), "中风险")
        self.assertEqual(calculate_risk_level(75), "高风险")
        self.assertEqual(calculate_risk_level(95), "极高风险")


if __name__ == "__main__":
    unittest.main()
