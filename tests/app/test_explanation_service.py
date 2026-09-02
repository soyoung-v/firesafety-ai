"""app/service/explanation_service.py 단위 테스트 - provider는 fake로 대체, 실제 LLM 호출 없음."""

from __future__ import annotations

import pytest

from app.config import LlmSettings
from app.schema.explain import ExplainRequest, PredictionEvidence, SensorEvidence
from app.service import explanation_service
from app.service.explanation_provider import ExplanationProvider


class _FakeProvider(ExplanationProvider):
    def __init__(self, summary: str = "요약 문장입니다.", captured: dict | None = None):
        self._summary = summary
        self._captured = captured

    def generate_summary(self, system_prompt: str, user_prompt: str) -> str:
        if self._captured is not None:
            self._captured["system_prompt"] = system_prompt
            self._captured["user_prompt"] = user_prompt
        return self._summary


def _settings(enabled: bool = True) -> LlmSettings:
    return LlmSettings(enabled=enabled, provider="openai", openai_api_key="sk-test", openai_model="gpt-4o-mini",
                        timeout_seconds=5.0)


def _request() -> ExplainRequest:
    return ExplainRequest(
        circuit=1,
        prediction=PredictionEvidence(pred=0, proba=0.08, riskLevel="DANGER", riskScore=0.91),
        sensorEvidence=SensorEvidence(current=10.9),
    )


def test_generate_explanation_returns_provider_summary():
    summary = explanation_service.generate_explanation(_settings(), _FakeProvider("요약1"), _request())
    assert summary == "요약1"


def test_generate_explanation_raises_disabled_error_when_settings_disabled():
    with pytest.raises(explanation_service.ExplanationDisabledError):
        explanation_service.generate_explanation(_settings(enabled=False), _FakeProvider(), _request())


def test_generate_explanation_raises_config_error_when_provider_missing():
    with pytest.raises(explanation_service.ExplanationConfigError):
        explanation_service.generate_explanation(_settings(), None, _request())


def test_generate_explanation_passes_evidence_derived_prompt_to_provider():
    captured: dict = {}
    explanation_service.generate_explanation(_settings(), _FakeProvider(captured=captured), _request())

    assert "DANGER" in captured["user_prompt"]
    assert "0.91" in captured["user_prompt"]


def test_system_prompt_contains_core_safety_rules():
    captured: dict = {}
    explanation_service.generate_explanation(_settings(), _FakeProvider(captured=captured), _request())

    system_prompt = captured["system_prompt"]
    assert "변경하거나 새로 계산하지 않는다" in system_prompt
    assert "화재 확률" in system_prompt
    assert "임의로 만들어내지 않는다" in system_prompt
