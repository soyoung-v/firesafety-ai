"""OpenAiExplanationProvider 단위 테스트. 실제 OpenAI API를 호출하지 않고 SDK 클라이언트를 stub한다."""

from __future__ import annotations

from types import SimpleNamespace

import openai
import pytest

from app.service.explanation_provider import ProviderError, ProviderResponseError, ProviderTimeoutError
from app.service.openai_explanation_provider import OpenAiExplanationProvider


class _StubCompletions:
    def __init__(self, result=None, exc: Exception | None = None):
        self._result = result
        self._exc = exc

    def parse(self, **kwargs):
        if self._exc is not None:
            raise self._exc
        return self._result


def _make_provider(completions: _StubCompletions) -> OpenAiExplanationProvider:
    provider = OpenAiExplanationProvider(api_key="sk-test-not-real", model="gpt-4o-mini", timeout_seconds=1.0)
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return provider


def _parsed_completion(analysis_summary: str, refusal: str | None = None):
    parsed = SimpleNamespace(analysisSummary=analysis_summary) if analysis_summary is not None else None
    message = SimpleNamespace(parsed=parsed, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_generate_summary_returns_parsed_text():
    completion = _parsed_completion("최근 전류 증가가 관찰됩니다.")
    provider = _make_provider(_StubCompletions(result=completion))

    summary = provider.generate_summary("system", "user")

    assert summary == "최근 전류 증가가 관찰됩니다."


def test_generate_summary_raises_provider_timeout_error():
    provider = _make_provider(_StubCompletions(exc=openai.APITimeoutError(request=SimpleNamespace())))

    with pytest.raises(ProviderTimeoutError):
        provider.generate_summary("system", "user")


def test_generate_summary_raises_provider_error_on_connection_failure():
    provider = _make_provider(_StubCompletions(exc=openai.APIConnectionError(request=SimpleNamespace())))

    with pytest.raises(ProviderError):
        provider.generate_summary("system", "user")


def test_generate_summary_raises_provider_response_error_when_parsed_missing():
    completion = _parsed_completion(None)
    provider = _make_provider(_StubCompletions(result=completion))

    with pytest.raises(ProviderResponseError):
        provider.generate_summary("system", "user")


def test_generate_summary_raises_provider_response_error_on_refusal():
    completion = _parsed_completion("무시될 값", refusal="정책상 응답할 수 없습니다")
    provider = _make_provider(_StubCompletions(result=completion))

    with pytest.raises(ProviderResponseError):
        provider.generate_summary("system", "user")


def test_generate_summary_raises_provider_response_error_on_blank_summary():
    completion = _parsed_completion("   ")
    provider = _make_provider(_StubCompletions(result=completion))

    with pytest.raises(ProviderResponseError):
        provider.generate_summary("system", "user")
