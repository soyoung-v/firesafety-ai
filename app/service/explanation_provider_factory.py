"""LlmSettings -> ExplanationProvider 생성. provider 종류 분기는 여기 한 곳에서만 한다."""

from __future__ import annotations

from app.config import LlmSettings

from .explanation_provider import ExplanationProvider
from .openai_explanation_provider import OpenAiExplanationProvider


# 설정이 비활성화됐거나 필요한 값(API Key 등)이 없으면 None을 반환한다 - 호출부가 이 경우를 명시적으로 처리한다.
def build_provider(settings: LlmSettings) -> ExplanationProvider | None:
    if not settings.enabled:
        return None
    if settings.provider == "openai":
        if not settings.openai_api_key:
            return None
        return OpenAiExplanationProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            timeout_seconds=settings.timeout_seconds,
        )
    return None
