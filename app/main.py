"""FastAPI 앱 진입점. startup 시 4개 모델을 1회 로드해 요청마다 다시 읽지 않는다."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api import explain, health, model_info, predict
from app.config import load_llm_settings
from app.service.explanation_provider_factory import build_provider
from app.service.model_registry import ModelRegistry


# startup에서 모델 전체 로드, shutdown에서는 별도 정리가 필요 없음(파일 핸들을 들고 있지 않음)
# LLM 설명 provider는 ML 모델과 달리 로드 실패/미설정이어도 서비스 시작을 막지 않는다(/explain만 503).
@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry()
    registry.load_all()
    app.state.model_registry = registry

    llm_settings = load_llm_settings()
    app.state.llm_settings = llm_settings
    app.state.explanation_provider = build_provider(llm_settings)

    yield


app = FastAPI(
    title="ArcGuard AI Service",
    version="1.0.0",
    description=(
        "ArcGuard 전기설비 센서 데이터를 분석하는 Synthetic Dataset 기반 포트폴리오 AI 서비스입니다. "
        "실제 전기화재 탐지/예방 성능이 검증된 것이 아닙니다. "
        "기존 Spring Boot `/predict` 계약(legacy)을 그대로 지원하며, "
        "optional `context`를 함께 보내면 riskLevel/riskScore/anomaly/anomalyScore도 계산합니다."
    ),
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(model_info.router)
app.include_router(predict.router)
app.include_router(explain.router)


# Pydantic 검증 오류의 기본 응답은 거절된 원본 값을 그대로 echo하는데, NaN/Infinity처럼 표준
# JSON으로 인코딩 불가능한 값이 입력되면 응답 자체가 인코딩 실패로 죽는다 - 원본 값(input/ctx)을
# 빼고 type/loc/msg만 돌려주는 안전한 422 응답으로 통일한다. 내부 traceback도 노출하지 않는다.
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    safe_errors = [
        {"type": error.get("type"), "loc": error.get("loc"), "msg": error.get("msg")}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": safe_errors})
