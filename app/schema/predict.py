"""POST /predict 요청/응답 스키마.

기존 Spring Boot 계약(m_no/circuits/circuit/samples/am/count, threshold/results/circuit/proba/
pred/n_samples/warning)은 필드명·타입·의미를 그대로 유지한다. `context`는 이번 Phase에서 새로 추가한
optional 확장이며, 없어도 기존 요청은 그대로 동작한다(ARC 판정만 수행).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SampleRequest(BaseModel):
    """회로 샘플 1건 - 기존 계약 그대로."""

    am: float = Field(..., ge=0, allow_inf_nan=False, description="전류값(A)")
    count: int = Field(..., ge=0, description="회로 카운터. 0=정상, 증가 시 아크 의심")


class CircuitRequest(BaseModel):
    """회로 1개 요청 - 기존 계약 그대로."""

    circuit: int = Field(..., ge=1, le=10, description="회로 번호(1~10)")
    samples: list[SampleRequest] = Field(..., description="회로 샘플 목록. 최소 30개 이상 권장")


class ContextSampleRequest(BaseModel):
    """분전반/장비 공통 센서값 1개 프레임분.

    같은 요청의 각 circuit.samples와 같은 개수·순서(프레임 정렬)여야 Risk/Anomaly 계산에 사용된다.
    일부 필드가 비어 있으면(leakage_current/temperature/total_current 중 하나라도 None) 해당
    요청은 Risk/Anomaly 계산에서 제외된다 - 임의 기본값을 채우지 않는다.
    """

    voltage: Optional[float] = Field(None, ge=0, allow_inf_nan=False)
    leakage_current: Optional[float] = Field(None, ge=0, allow_inf_nan=False)
    temperature: Optional[float] = Field(None, allow_inf_nan=False)
    humidity: Optional[float] = Field(None, ge=0, le=100, allow_inf_nan=False)
    fire_raw: Optional[float] = Field(None, ge=0, allow_inf_nan=False)
    gas_raw: Optional[float] = Field(None, ge=0, allow_inf_nan=False)
    door_open: Optional[bool] = None
    total_current: Optional[float] = Field(None, ge=0, allow_inf_nan=False)
    total_power: Optional[float] = Field(None, ge=0, allow_inf_nan=False)


class ContextRequest(BaseModel):
    """신규 확장(optional) - 기존 계약에는 없던 필드. Risk Classifier/Anomaly Detector 계산에 사용된다."""

    samples: list[ContextSampleRequest] = Field(..., description="circuit.samples와 프레임 정렬된 시계열")


class PredictRequest(BaseModel):
    """POST /predict 요청 - 기존 Spring Boot AiPredictionReq와 동일 구조 + optional context."""

    m_no: str = Field(..., description="분전반 장비번호")
    circuits: list[CircuitRequest] = Field(..., description="회로별 데이터 목록. 최대 10회로")
    context: Optional[ContextRequest] = Field(
        None,
        description="신규 확장(optional). 없으면 riskLevel/riskScore/anomaly/anomalyScore는 null로 반환된다.",
    )

    model_config = {
        "json_schema_extra": {
            "description": (
                "Synthetic Dataset 기반 포트폴리오 AI 서비스의 예측 요청입니다. "
                "context 없이 m_no/circuits만 보내는 기존 형태(legacy)도 그대로 지원합니다."
            ),
            "examples": [
                {
                    "summary": "legacy request (기존 Spring Boot 계약, context 없음)",
                    "value": {
                        "m_no": "00001",
                        "circuits": [{"circuit": 1, "samples": [{"am": 5.2, "count": 0}]}],
                    },
                },
                {
                    "summary": "extended request (context 포함, Risk/Anomaly까지 계산)",
                    "value": {
                        "m_no": "00001",
                        "circuits": [{"circuit": 1, "samples": [{"am": 5.2, "count": 0}]}],
                        "context": {
                            "samples": [
                                {
                                    "voltage": 224.0,
                                    "leakage_current": 2.0,
                                    "temperature": 27.2,
                                    "humidity": 48.4,
                                    "fire_raw": 500,
                                    "gas_raw": 500,
                                    "door_open": False,
                                    "total_current": 10.0,
                                    "total_power": 2000,
                                }
                            ]
                        },
                    },
                },
            ],
        }
    }


class CircuitPredictionResult(BaseModel):
    """회로별 판정 결과 - 기존 5개 필드(circuit/proba/pred/n_samples/warning) + 신규 확장 5개 필드."""

    circuit: int
    proba: float
    pred: int
    n_samples: int
    warning: Optional[str] = None

    riskLevel: Optional[str] = None
    riskScore: Optional[float] = None
    anomaly: Optional[bool] = None
    anomalyScore: Optional[float] = None
    predictedCurrent: Optional[float] = None


class PredictResponse(BaseModel):
    """POST /predict 응답 - 기존 계약 그대로(m_no/threshold/results)."""

    m_no: str
    threshold: float
    results: list[CircuitPredictionResult]
