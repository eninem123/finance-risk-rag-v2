from unittest.mock import MagicMock, patch

import pytest

from src.finance_risk_rag.llm import LLMClientWrapper, LLMError


def test_llm_retry_logic():
    mock_client = MagicMock()
    # Mock behavior: fail twice then succeed
    mock_client.chat.completions.create.side_effect = [
        Exception("API Error"),
        Exception("Rate Limit"),
        MagicMock(choices=[MagicMock(message=MagicMock(content="Success"))]),
    ]

    with patch("src.finance_risk_rag.llm.OpenAI", return_value=mock_client):
        wrapper = LLMClientWrapper(api_key="test")
        # Set short initial backoff for fast testing
        result = wrapper.chat([{"role": "user", "content": "hi"}], initial_backoff=0.01)
        assert result == "Success"
        assert mock_client.chat.completions.create.call_count == 3


def test_llm_failure_after_retries():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("Permanent Error")

    with patch("src.finance_risk_rag.llm.OpenAI", return_value=mock_client):
        wrapper = LLMClientWrapper(api_key="test")
        with pytest.raises(LLMError, match="LLM call failed after 3 retries"):
            wrapper.chat([{"role": "user", "content": "hi"}], initial_backoff=0.01)
