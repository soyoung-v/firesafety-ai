"""app/config.py::load_llm_settings 단위 테스트. 실제 OpenAI를 호출하지 않는다."""

from __future__ import annotations

from app.config import DEFAULT_OPENAI_MODEL, DEFAULT_TIMEOUT_SECONDS, load_llm_settings


def test_default_settings_are_disabled(monkeypatch):
    # 개발자 로컬 .env에 실제 값(LLM_EXPLANATION_ENABLED=true, OPENAI_MODEL 등)이 들어있어도
    # 이 테스트는 "값이 전혀 없을 때의 기본값"만 검증해야 하므로 관련 env 전부를 명시적으로 지운다
    # (ambient 환경에 의존하지 않는 격리 - Phase 15 test-isolation 수정).
    for key in ("LLM_EXPLANATION_ENABLED", "LLM_PROVIDER", "OPENAI_API_KEY", "OPENAI_MODEL", "LLM_TIMEOUT_SECONDS"):
        monkeypatch.delenv(key, raising=False)

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
