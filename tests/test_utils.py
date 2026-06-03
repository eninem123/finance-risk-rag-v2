from src.finance_risk_rag.utils import clean_text, split_text_by_sentence, calculate_risk_level

def test_clean_text():
    text = "  Hello   World！。，；  "
    cleaned = clean_text(text)
    assert cleaned == "Hello World!.,;"

def test_split_text_by_sentence():
    text = "Sentence one. Sentence two? Sentence three! Long sentence that should not be split yet."
    sentences = split_text_by_sentence(text, max_len=20)
    assert len(sentences) >= 3
    assert "Sentence one." in sentences[0]

def test_calculate_risk_level():
    assert calculate_risk_level(20) == "Low Risk"
    assert calculate_risk_level(50) == "Medium Risk"
    assert calculate_risk_level(80) == "High Risk"
    assert calculate_risk_level(95) == "Extreme Risk"
