"""POST /explain - 이미 계산된 AI 진단 결과를 사람이 읽을 한국어 설명으로 바꾼다.

/predict의 핵심 추론 경로와 완전히 분리된 별도 엔드포인트다 - 외부 LLM API 장애/지연이 ML 판정에
영향을 주지 않는다. 실패 사유별로 명확한 상태코드를 반환하고, 내부 traceback/API Key/요청 payload는
응답에도 로그에도 남기지 않는다.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request

from app.schema.explain import ExplainRequest, ExplainResponse
from app.service import explanation_service
from app.service.explanation_provider import ProviderError, ProviderResponseError, ProviderTimeoutError

router = APIRouter()
logger = logging.getLogger("app.explain")


@router.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest, http_request: Request) -> ExplainResponse:
    settings = http_request.app.state.llm_settings
    provider = http_request.app.state.explanation_provider

    started = time.monotonic()
    try:
        summary = explanation_service.generate_explanation(settings, provider, request)
    except explanation_service.ExplanationDisabledError as exc:
        logger.info("explain 요청 거절 - LLM 설명 기능 비활성화")
        raise HTTPException(status_code=503, detail="LLM 설명 기능이 비활성화되어 있습니다") from exc
    except explanation_service.ExplanationConfigError as exc:
        logger.warning("explain 요청 거절 - provider 미설정")
        raise HTTPException(status_code=503, detail="LLM provider가 설정되지 않았습니다") from exc
    except ProviderTimeoutError as exc:
        logger.warning("explain 실패 - timeout, provider=%s", settings.provider)
        raise HTTPException(status_code=504, detail="LLM 응답이 지연되어 실패했습니다") from exc
    except ProviderResponseError as exc:
        logger.warning("explain 실패 - malformed response, provider=%s", settings.provider)
        raise HTTPException(status_code=502, detail="LLM 응답 형식이 올바르지 않습니다") from exc
    except ProviderError as exc:
        logger.warning("explain 실패 - provider error, provider=%s", settings.provider)
        raise HTTPException(status_code=502, detail="LLM 제공자 호출에 실패했습니다") from exc

    elapsed_ms = int((time.monotonic() - started) * 1000)
    logger.info("explain 요청 성공 - provider=%s, latency_ms=%d", settings.provider, elapsed_ms)
    return ExplainResponse(analysisSummary=summary)
