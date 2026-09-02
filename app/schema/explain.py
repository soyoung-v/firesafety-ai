"""POST /explain 요청/응답 스키마.

이미 계산된 AI 진단 결과(ML)를 사람이 읽을 한국어 설명으로 바꾸는 것이 목적이다. 원시 센서
샘플 배열을 다시 받지 않는다 - /predict가 이미 계산을 마친 값(prediction)과 그 근거로 쓸 수 있는
축약된 스냅샷(sensorEvidence/trendEvidence)만 받는다. Synthetic Dataset의 정답 라벨(scenario,
risk_level ground truth, run_id 등)을 담을 필드는 이 스키마에 존재하지 않는다 - 애초에 받을 수 없다.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictionEvidence(BaseModel):
    """이미 확정된 ML 판정 결과 - LLM은 이 값을 재계산하거나 바꾸지 않는다. /predict 응답과 동일한 의미."""

    # extra="forbid": scenario/risk_level ground truth/run_id 같은 Synthetic Dataset 정답 필드가
    # 실수로 섞여 들어와도 조용히 무시되지 않고 요청 자체가 422로 거절된다.
    model_config = {"extra": "forbid"}

    pred: int = Field(..., ge=0, le=1, description="0=NORMAL, 1=ARC (Legacy ARC Classifier)")
    proba: float = Field(..., ge=0, le=1)
    riskLevel: Optional[str] = Field(None, description="NORMAL/WARNING/DANGER")
    riskScore: Optional[float] = Field(None, ge=0, le=1)
    anomaly: Optional[bool] = None
    anomalyScore: Optional[float] = Field(None, ge=0, le=1)
    predictedCurrent: Optional[float] = Field(None, description="다음 sample 시점의 예측 전류값(A)")


class SensorEvidence(BaseModel):
    """설명 생성에 필요한 최신 센서값 스냅샷 - 원시 시계열 전체가 아니다."""

    model_config = {"extra": "forbid"}

    current: Optional[float] = Field(None, ge=0)
    arcCount: Optional[int] = Field(None, ge=0)
    temperature: Optional[float] = None
    leakageCurrent: Optional[float] = Field(None, ge=0)
    totalCurrent: Optional[float] = Field(None, ge=0)


class TrendEvidence(BaseModel):
    """추세 근거(optional) - 호출자가 이미 계산해 둔 feature 값만 받는다(원시 샘플 재연산 안 함)."""

    model_config = {"extra": "forbid"}

    currentSlope: Optional[float] = None
    temperatureSlope: Optional[float] = None


class ExplainRequest(BaseModel):
    """POST /explain 요청."""

    circuit: int = Field(..., ge=1, le=10)
    prediction: PredictionEvidence
    sensorEvidence: SensorEvidence
    trendEvidence: Optional[TrendEvidence] = None

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "circuit": 1,
                    "prediction": {
                        "pred": 0,
                        "proba": 0.08,
                        "riskLevel": "WARNING",
                        "riskScore": 0.63,
                        "anomaly": True,
                        "anomalyScore": 0.78,
                        "predictedCurrent": 12.4,
                    },
                    "sensorEvidence": {
                        "current": 10.9,
                        "arcCount": 0,
                        "temperature": 72.3,
                        "leakageCurrent": 11.0,
                        "totalCurrent": 18.2,
                    },
                    "trendEvidence": {"currentSlope": 0.12, "temperatureSlope": 0.81},
                }
            ]
        }
    }


class ExplainResponse(BaseModel):
    """POST /explain 응답 - 최소 안정 계약은 analysisSummary 하나뿐이다."""

    analysisSummary: str = Field(..., min_length=1)
