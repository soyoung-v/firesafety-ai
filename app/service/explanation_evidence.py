"""ExplainRequest(이미 계산된 ML 결과 + 축약 센서 스냅샷) -> LLM 프롬프트용 Evidence dict.

deterministic한 순수 함수만 둔다 - LLM 호출과 완전히 분리되어 있어 단위 테스트가 쉽다.
Synthetic Dataset의 정답 라벨(scenario/risk_level ground truth/run_id/aerror 등)은 애초에
ExplainRequest 스키마에 필드가 없어 여기서 다룰 수도 없다.
"""

from __future__ import annotations

from app.schema.explain import ExplainRequest


# None 값은 생략한다 - 모르는 값을 0 등 가짜 값으로 채우지 않는다
def _drop_none(values: dict) -> dict:
    return {key: value for key, value in values.items() if value is not None}


# ExplainRequest -> LLM 프롬프트에 그대로 넣을 evidence dict
def build_evidence(request: ExplainRequest) -> dict:
    evidence: dict = {
        "circuit": request.circuit,
        "prediction": _drop_none(request.prediction.model_dump()),
        "sensorEvidence": _drop_none(request.sensorEvidence.model_dump()),
    }
    if request.trendEvidence is not None:
        trend = _drop_none(request.trendEvidence.model_dump())
        if trend:
            evidence["trendEvidence"] = trend
    return evidence
