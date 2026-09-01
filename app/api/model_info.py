"""GET /model-info - 4개 모델의 공개 가능한 정보를 반환한다 (로컬 경로/내부 구조는 노출하지 않음)."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()

NOTE = "Synthetic Dataset 기반 포트폴리오 모델입니다. 실제 전기화재 탐지/예방 성능이 검증된 것이 아닙니다."


# 모델별 공개 정보(타입/feature 개수/window/threshold/labels/생성일)를 반환
@router.get("/model-info")
def model_info(request: Request) -> dict:
    registry = request.app.state.model_registry
    return {"note": NOTE, "models": registry.public_info()}
