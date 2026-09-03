"""POST /explain 통합 테스트. 실제 OpenAI를 호출하지 않고 app.state.explanation_provider를 fake로 교체한다.

/explain 전용으로 함수 스코프 TestClient를 쓴다(conftest의 module-scope client와 상태를 공유하지
않기 위해) - 각 테스트가 app.state.llm_settings/explanation_provider를 독립적으로 설정한다.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import LlmSettings
from app.main import app
from app.service.explanation_provider import ExplanationProvider, ProviderError, ProviderResponseError, ProviderTimeoutError


class _FakeProvider(ExplanationProvider):
    def __init__(self, summary: str = "정상 요약입니다."):
        self._summary = summary

    def generate_summary(self, system_prompt: str, user_prompt: str) -> str:
        return self._summary


class _FailingProvider(ExplanationProvider):
    def __init__(self, exc: Exception):
        self._exc = exc

    def generate_summary(self, system_prompt: str, user_prompt: str) -> str:
        raise self._exc


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _payload(**overrides) -> dict:
    payload = {
        "circuit": 1,
        "prediction": {"pred": 0, "proba": 0.08, "riskLevel": "WARNING", "riskScore": 0.63,
                        "anomaly": True, "anomalyScore": 0.78, "predictedCurrent": 12.4},
        "sensorEvidence": {"current": 10.9, "arcCount": 0, "temperature": 72.3,
                            "leakageCurrent": 11.0, "totalCurrent": 18.2},
    }
    payload.update(overrides)
    return payload


def _enable(client: TestClient, provider: ExplanationProvider | None) -> None:
    client.app.state.llm_settings = LlmSettings(
        enabled=True, provider="openai", openai_api_key="sk-test", openai_model="gpt-4o-mini", timeout_seconds=5.0
    )
    client.app.state.explanation_provider = provider


def test_explain_disabled_by_default_returns_503(client):
    # lifespan은 실제 환경변수로 로드되므로, 개발자 로컬 .env에 실제 값이 있어도 이 테스트가
    # "비활성화 상태"를 검증하도록 app.state를 명시적으로 비활성화로 고정한다(ambient 환경에
    # 의존하지 않는 격리 - Phase 15 test-isolation 수정).
    client.app.state.llm_settings = LlmSettings(
        enabled=False, provider="openai", openai_api_key=None, openai_model="gpt-4o-mini", timeout_seconds=5.0
    )
    client.app.state.explanation_provider = None

    response = client.post("/explain", json=_payload())
    assert response.status_code == 503


def test_explain_enabled_but_no_provider_returns_503(client):
    client.app.state.llm_settings = LlmSettings(
        enabled=True, provider="openai", openai_api_key=None, openai_model="gpt-4o-mini", timeout_seconds=5.0
    )
    client.app.state.explanation_provider = None

    response = client.post("/explain", json=_payload())
    assert response.status_code == 503


def test_explain_success_returns_analysis_summary(client):
    _enable(client, _FakeProvider("최근 전류 상승과 이상 패턴이 함께 감지되었습니다."))

    response = client.post("/explain", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert body == {"analysisSummary": "최근 전류 상승과 이상 패턴이 함께 감지되었습니다."}
    # 응답 계약에 riskLevel 등 ML 필드가 아예 존재하지 않는다 - LLM이 구조적으로 바꿀 수 없다
    assert set(body.keys()) == {"analysisSummary"}


def test_explain_provider_timeout_returns_504(client):
    _enable(client, _FailingProvider(ProviderTimeoutError("timeout")))
    response = client.post("/explain", json=_payload())
    assert response.status_code == 504
    assert response.json() == {"detail": "LLM 응답이 지연되어 실패했습니다"}


def test_explain_provider_malformed_response_returns_502(client):
    _enable(client, _FailingProvider(ProviderResponseError("malformed")))
    response = client.post("/explain", json=_payload())
    assert response.status_code == 502


def test_explain_provider_generic_error_returns_502(client):
    _enable(client, _FailingProvider(ProviderError("boom")))
    response = client.post("/explain", json=_payload())
    assert response.status_code == 502


def test_explain_error_response_does_not_leak_internals(client):
    _enable(client, _FailingProvider(ProviderError("sk-should-not-leak internal traceback details")))
    response = client.post("/explain", json=_payload())
    assert "sk-should-not-leak" not in response.text
    assert "traceback" not in response.text.lower()


def test_explain_rejects_missing_required_fields(client):
    response = client.post("/explain", json={"circuit": 1})
    assert response.status_code == 422


def test_predict_still_works_when_explanation_provider_is_broken(client):
    # /predict가 LLM 상태와 완전히 독립적으로 동작하는지 확인
    class _AlwaysFailsProvider(ExplanationProvider):
        def generate_summary(self, system_prompt: str, user_prompt: str) -> str:
            raise ProviderError("LLM 완전히 죽음")

    _enable(client, _AlwaysFailsProvider())

    samples = [{"am": 5.2, "count": 0}] * 35
    response = client.post("/predict", json={"m_no": "00001", "circuits": [{"circuit": 1, "samples": samples}]})

    assert response.status_code == 200
    assert response.json()["results"][0]["pred"] in (0, 1)
