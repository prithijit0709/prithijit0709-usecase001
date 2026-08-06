from unittest.mock import MagicMock, patch


def test_groq_factory_uses_shared_rate_limiter_and_config():
    from core import llm

    with patch("core.llm.ChatGroq", return_value=MagicMock()) as chat_groq:
        llm.make_llm()
        llm.make_llm()

    first = chat_groq.call_args_list[0].kwargs
    second = chat_groq.call_args_list[1].kwargs
    assert first["model"] == "llama-3.1-8b-instant"
    assert first["temperature"] == 0
    assert first["max_tokens"] == 1200
    assert first["max_retries"] == 4
    assert first["rate_limiter"] is second["rate_limiter"]


def test_groq_rate_limiter_uses_configured_interval():
    from core import llm

    expected_rate = 1 / 65
    assert llm.GROQ_RATE_LIMITER.requests_per_second == expected_rate