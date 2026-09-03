"""대표 시나리오(NORMAL/OVERHEATING형/ARC) fixture로 prompt 구성의 구조적 제약을 검증한다.

실제 provider 문장을 고정(golden text)하지 않는다 - LLM 출력은 provider가 생성하므로 여기서는
'evidence가 ML 결과를 그대로 담는지', 'LLM에게 강제 규칙이 전달되는지', 'ARC로 오인될 값이 섞이지
않는지'처럼 구조/제약만 확인한다.
"""

from __future__ import annotations

from app.schema.explain import ExplainRequest, PredictionEvidence, SensorEvidence, TrendEvidence
from app.service.explanation_evidence import build_evidence
from app.service.explanation_prompt import SYSTEM_PROMPT, build_user_prompt

NORMAL_REQUEST = ExplainRequest(
    circuit=1,
    prediction=PredictionEvidence(pred=0, proba=0.02, riskLevel="NORMAL", riskScore=0.05,
                                   anomaly=False, anomalyScore=0.1, predictedCurrent=5.1),
    sensorEvidence=SensorEvidence(current=5.0, arcCount=0, temperature=27.1,
                                   leakageCurrent=2.0, totalCurrent=10.0),
)

OVERHEATING_LIKE_REQUEST = ExplainRequest(
    circuit=5,
    prediction=PredictionEvidence(pred=0, proba=0.03, riskLevel="DANGER", riskScore=0.94,
                                   anomaly=True, anomalyScore=0.88, predictedCurrent=5.2),
    sensorEvidence=SensorEvidence(current=5.1, arcCount=0, temperature=85.4,
                                   leakageCurrent=2.1, totalCurrent=10.3),
    trendEvidence=TrendEvidence(currentSlope=0.01, temperatureSlope=1.2),
)

ARC_REQUEST = ExplainRequest(
    circuit=3,
    prediction=PredictionEvidence(pred=1, proba=0.98, riskLevel="DANGER", riskScore=0.97,
                                   anomaly=True, anomalyScore=0.95, predictedCurrent=6.0),
    sensorEvidence=SensorEvidence(current=4.3, arcCount=12, temperature=27.5,
                                   leakageCurrent=2.0, totalCurrent=10.1),
)


def test_normal_case_evidence_reflects_low_risk_as_is():
    evidence = build_evidence(NORMAL_REQUEST)
    assert evidence["prediction"]["riskLevel"] == "NORMAL"
    assert evidence["prediction"]["pred"] == 0


def test_overheating_like_case_pred_stays_zero_not_forced_to_arc():
    # pred=0(ARC 아님)인데 riskLevel=DANGER인 경우 - evidence가 이 둘을 섞지 않고 그대로 보존해야 한다
    evidence = build_evidence(OVERHEATING_LIKE_REQUEST)
    assert evidence["prediction"]["pred"] == 0
    assert evidence["prediction"]["riskLevel"] == "DANGER"
    assert evidence["sensorEvidence"]["arcCount"] == 0


def test_arc_case_evidence_carries_arc_count_and_pred():
    evidence = build_evidence(ARC_REQUEST)
    assert evidence["prediction"]["pred"] == 1
    assert evidence["sensorEvidence"]["arcCount"] == 12


def test_user_prompt_contains_only_evidence_derived_from_request():
    evidence = build_evidence(OVERHEATING_LIKE_REQUEST)
    user_prompt = build_user_prompt(evidence)

    assert "85.4" in user_prompt  # temperature
    assert "DANGER" in user_prompt
    # Synthetic 정답 라벨은애초에 evidence에 없으므로 prompt에도 없다
    for forbidden in ("OVERHEATING", "scenario", "run_id", "ground_truth"):
        assert forbidden not in user_prompt


def test_system_prompt_forbids_forced_arc_labeling_and_fire_probability_language():
    assert "새로 계산하지 않는다" in SYSTEM_PROMPT
    assert "화재 확률" in SYSTEM_PROMPT
    assert "가스/불꽃" in SYSTEM_PROMPT
    assert "시간 단위로 표현하지 않는다" in SYSTEM_PROMPT


def test_system_prompt_instructs_natural_language_over_raw_field_names():
    # 필드명을 그대로 나열하지 말라는 지시와, 무엇으로 바꿔 써야 하는지 매핑이 둘 다 있어야 한다
    assert "필드명을 문장에 그대로 나열하지" in SYSTEM_PROMPT
    assert "종합 위험도" in SYSTEM_PROMPT
    assert "아크 판정" in SYSTEM_PROMPT
    assert "이상 패턴" in SYSTEM_PROMPT
    assert "예상 전류" in SYSTEM_PROMPT


def test_system_prompt_instructs_selecting_only_key_sensor_evidence():
    assert "2~3개만" in SYSTEM_PROMPT


def test_system_prompt_instructs_output_structure_order():
    assert "현재 판정" in SYSTEM_PROMPT
    assert "주의해야 할 근거" in SYSTEM_PROMPT
    assert "확인 권장사항" in SYSTEM_PROMPT


def test_system_prompt_instructs_percent_confidence_and_one_decimal_numbers():
    assert "%로 표현" in SYSTEM_PROMPT
    assert "소수점 1자리까지만" in SYSTEM_PROMPT
