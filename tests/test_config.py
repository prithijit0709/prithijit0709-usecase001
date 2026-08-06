import importlib

import core.config as config


def test_model_names_are_loaded_from_environment(monkeypatch):
    original_groq_model = config.GROQ_MODEL
    original_gemini_model = config.GEMINI_MODEL
    monkeypatch.setenv("GROQ_MODEL", "test-groq-model")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")

    try:
        reloaded_config = importlib.reload(config)

        assert reloaded_config.GROQ_MODEL == "test-groq-model"
        assert reloaded_config.GEMINI_MODEL == "gemini-test-model"
    finally:
        monkeypatch.setenv("GROQ_MODEL", original_groq_model)
        monkeypatch.setenv("GEMINI_MODEL", original_gemini_model)
        importlib.reload(config)


def test_groq_runtime_defaults(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "")
    monkeypatch.setenv("GROQ_MIN_REQUEST_INTERVAL_SECONDS", "65")
    monkeypatch.setenv("GROQ_MAX_RETRIES", "4")
    monkeypatch.setenv("GROQ_MAX_OUTPUT_TOKENS", "1200")

    reloaded_config = importlib.reload(config)

    assert reloaded_config.GROQ_MODEL == "llama-3.1-8b-instant"
    assert reloaded_config.GROQ_MIN_REQUEST_INTERVAL_SECONDS == 65.0
    assert reloaded_config.GROQ_MAX_RETRIES == 4
    assert reloaded_config.GROQ_MAX_OUTPUT_TOKENS == 1200