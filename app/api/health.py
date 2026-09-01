"""GET /health - 기존 status 필드를 유지하며 모델별 로딩 상태를 추가한다."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


# 모델 로딩 상태 기준으로 서비스 상태를 반환 (기존 status 필드 유지, models는 신규 확장)
@router.get("/health")
def health(request: Request) -> dict:
    registry = getattr(request.app.state, "model_registry", None)
    if registry is None:
        return {"status": "UP", "models": None}

    model_status = registry.status()
    overall = "UP" if all(model_status.values()) else "DEGRADED"
    return {"status": overall, "models": model_status}
