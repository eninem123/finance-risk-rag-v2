from src.finance_risk_rag.utils import calculate_risk_level, clean_text, split_text_by_sentence


def test_clean_text():
    text = "  Hello \n World! 3。5  ，"
    cleaned = clean_text(text)
    assert cleaned == "Hello World! 3.5 ,"


def test_split_text_by_sentence():
    text = "这是第一个句子。这是第二个句子！这是第三个吗？"
    # Set max_len small to prevent merging for test purposes
    sentences = split_text_by_sentence(text, max_len=10, min_len=2)
    assert len(sentences) == 3
    assert "这是第一个句子。" in sentences


def test_calculate_risk_level():
    assert calculate_risk_level(20) == "低风险"
    assert calculate_risk_level(45) == "中风险"
    assert calculate_risk_level(75) == "高风险"
    assert calculate_risk_level(100) == "极高风险"
