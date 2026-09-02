"""GET /model-info - 4개 모델의 공개 가능한 정보를 반환한다 (로컬 경로/내부 구조는 노출하지 않음)."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()

NOTE = "Synthetic Dataset 기반 포트폴리오 모델입니다. 실제 전기화재 탐지/예방 성능이 검증된 것이 아닙니다."


# 모델별 공개 정보(타입/feature 개수/window/threshold/labels/생성일) + LLM provider 이름(설정 여부만)을 반환.
# LLM은 학습된 ML Artifact가 아니므로 models와 섞지 않고 별도 필드로 둔다. API Key/내부 endpoint는 노출하지 않는다.
@router.get("/model-info")
def model_info(request: Request) -> dict:
    registry = request.app.state.model_registry
    settings = request.app.state.llm_settings
    provider = request.app.state.explanation_provider

    explanation_provider_name = settings.provider if provider is not None else None
    return {"note": NOTE, "models": registry.public_info(), "explanationProvider": explanation_provider_name}
