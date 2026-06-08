import unittest
from finance_risk_rag.utils import clean_text, split_text_by_sentence, calculate_risk_level

class TestUtils(unittest.TestCase):
    def test_clean_text(self):
        text = "  Hello   World! 。 3。5 "
        # 原始代码中 clean_text 先将连续空格转为单个空格，包括句号前后的空格
        # "  Hello   World! 。 3。5 " -> "Hello World! 。 3.5" (3。5变3.5)
        # 然后 。 变 . -> "Hello World! . 3.5"
        expected = "Hello World! . 3.5"
        self.assertEqual(clean_text(text), expected)

    def test_split_text_by_sentence(self):
        text = "第一句。第二句！第三句？"
        # 默认 max_len=200, min_len=20. "第一句。第二句！第三句？" 长度较短，会被合并。
        # 如果要测试拆分，需要调小 max_len
        sentences = split_text_by_sentence(text, max_len=5, min_len=2)
        self.assertEqual(len(sentences), 3)
        self.assertIn("第一句。", sentences)

    def test_calculate_risk_level(self):
        self.assertEqual(calculate_risk_level(10), "低风险")
        self.assertEqual(calculate_risk_level(45), "中风险")
        self.assertEqual(calculate_risk_level(75), "高风险")
        self.assertEqual(calculate_risk_level(95), "极高风险")

if __name__ == '__main__':
    unittest.main()
