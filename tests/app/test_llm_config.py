"""app/config.py::load_llm_settings 단위 테스트. 실제 OpenAI를 호출하지 않는다."""

from __future__ import annotations

from app.config import DEFAULT_OPENAI_MODEL, DEFAULT_TIMEOUT_SECONDS, load_llm_settings


def test_default_settings_are_disabled(monkeypatch):
    monkeypatch.delenv("LLM_EXPLANATION_ENABLED", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = load_llm_settings()

    assert settings.enabled is False
    assert settings.provider == "openai"
    assert settings.openai_api_key is None
    assert settings.openai_model == DEFAULT_OPENAI_MODEL
    assert settings.timeout_seconds == DEFAULT_TIMEOUT_SECONDS


def test_explicit_enabled_true(monkeypatch):
    monkeypatch.setenv("LLM_EXPLANATION_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")

    settings = load_llm_settings()

    assert settings.enabled is True
    assert settings.openai_api_key == "sk-test-not-real"


def test_blank_env_values_fall_back_to_defaults(monkeypatch):
    # .env.example처럼 키만 있고 값이 빈 문자열인 경우도 "미설정"으로 취급해야 한다(안 그러면 float("") 등에서 죽음)
    monkeypatch.setenv("OPENAI_MODEL", "")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    settings = load_llm_settings()

    assert settings.openai_model == DEFAULT_OPENAI_MODEL
    assert settings.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert settings.openai_api_key is None


def test_custom_timeout_and_model(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "3.5")

    settings = load_llm_settings()

    assert settings.openai_model == "gpt-4o"
    assert settings.timeout_seconds == 3.5
