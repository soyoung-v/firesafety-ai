"""POST /predict - 기존 Spring Boot 계약(m_no/threshold/results)을 유지하며 신규 확장 필드를 추가한다."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schema.predict import PredictRequest, PredictResponse
from app.service import predict_service
from training.models.arc_metrics import THRESHOLD

router = APIRouter()


# circuit별 4-model 추론을 수행해 기존 계약 + 확장 필드를 함께 반환
@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, http_request: Request) -> PredictResponse:
    if not request.circuits:
        raise HTTPException(status_code=400, detail="circuits가 비어 있습니다")

    registry = http_request.app.state.model_registry
    context_samples = request.context.samples if request.context else None

    results = []
    for circuit_req in request.circuits:
        try:
            result = predict_service.predict_circuit(registry, circuit_req, context_samples)
        except predict_service.InsufficientSamplesError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        results.append(result)

    return PredictResponse(m_no=request.m_no, threshold=THRESHOLD, results=results)
