"""ExplainRequest(이미 계산된 ML 결과 + 축약 센서 스냅샷) -> LLM 프롬프트용 Evidence dict.

deterministic한 순수 함수만 둔다 - LLM 호출과 완전히 분리되어 있어 단위 테스트가 쉽다.
Synthetic Dataset의 정답 라벨(scenario/risk_level ground truth/run_id/aerror 등)은 애초에
ExplainRequest 스키마에 필드가 없어 여기서 다룰 수도 없다.

ML 원본 필드(pred/proba/riskLevel/riskScore/anomaly/anomalyScore/predictedCurrent 등)는 값을
바꾸지 않고 그대로 담는다 - "재계산 금지" 원칙은 evidence 단계에서도 지킨다. 여기서 하는 일은
딱 두 가지뿐이다: (1) 표시용 소수점 자리수를 정리하고(같은 값을 나타내는 표현만 바뀜),
(2) pred 방향에 맞춘 "판정 확신도(%)" 파생값을 추가한다(이미 있는 proba를 재계산하는 게
아니라, 사람이 오해하지 않게 방향만 맞춰 보여주는 것 - pred=0일 때 proba를 그대로 신뢰도로
읽으면 "정상인데 확신도 8%"처럼 거꾸로 읽힌다).
"""

from __future__ import annotations

from app.schema.explain import ExplainRequest

# 사람이 읽는 문장에 쓰기엔 과도하게 긴 소수(예: 27.218000411987305)를 그대로 넘기지 않도록
# 표시용 자리수만 잘라낸다 - 값 자체가 달라지는 반올림이 아니라 표현 정리다.
DISPLAY_DECIMALS = 1


# None 값은 생략한다 - 모르는 값을 0 등 가짜 값으로 채우지 않는다
def _drop_none(values: dict) -> dict:
    return {key: value for key, value in values.items() if value is not None}


def _round_display(values: dict, keys: tuple[str, ...]) -> dict:
    rounded = dict(values)
    for key in keys:
        value = rounded.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rounded[key] = round(float(value), DISPLAY_DECIMALS)
    return rounded


# ExplainRequest -> LLM 프롬프트에 그대로 넣을 evidence dict
def build_evidence(request: ExplainRequest) -> dict:
    prediction = _round_display(
        _drop_none(request.prediction.model_dump()), ("predictedCurrent",)
    )
    if "pred" in prediction and "proba" in prediction:
        # pred=1(ARC)이면 proba가 곧 그 판정의 확신도, pred=0(NORMAL)이면 (1-proba)가 확신도다.
        # proba 자체는 바꾸지 않고, 방향을 맞춘 파생값만 별도 키로 추가한다.
        proba = prediction["proba"]
        confidence = proba if prediction["pred"] == 1 else 1 - proba
        prediction["confidencePercent"] = round(confidence * 100)

    sensor_evidence = _round_display(
        _drop_none(request.sensorEvidence.model_dump()),
        ("current", "temperature", "leakageCurrent", "totalCurrent"),
    )

    evidence: dict = {
        "circuit": request.circuit,
        "prediction": prediction,
        "sensorEvidence": sensor_evidence,
    }
    if request.trendEvidence is not None:
        trend = _drop_none(request.trendEvidence.model_dump())
        if trend:
            evidence["trendEvidence"] = trend
    return evidence
