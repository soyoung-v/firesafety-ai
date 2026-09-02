"""Evidence Builder + Prompt + Provider를 묶어 analysisSummary를 생성한다.

/predict의 핵심 추론 경로와 완전히 분리되어 있다 - 이 모듈은 오직 POST /explain에서만 호출되며,
여기서 발생하는 실패는 ML 판정/저장에 어떤 영향도 주지 않는다.
"""

from __future__ import annotations

from app.config import LlmSettings
from app.schema.explain import ExplainRequest

from . import explanation_evidence, explanation_prompt
from .explanation_provider import ExplanationProvider


class ExplanationDisabledError(RuntimeError):
    """LLM_EXPLANATION_ENABLED=false."""


class ExplanationConfigError(RuntimeError):
    """기능은 켜져 있지만 provider를 구성할 값(API Key 등)이 없음."""


# 설정+provider+요청을 받아 analysisSummary 문자열을 생성한다. ML 결과 필드는 여기서 절대 계산하지 않는다.
def generate_explanation(
    settings: LlmSettings, provider: ExplanationProvider | None, request: ExplainRequest
) -> str:
    if not settings.enabled:
        raise ExplanationDisabledError("LLM 설명 기능이 비활성화되어 있습니다")
    if provider is None:
        raise ExplanationConfigError("LLM provider가 설정되지 않았습니다")

    evidence = explanation_evidence.build_evidence(request)
    user_prompt = explanation_prompt.build_user_prompt(evidence)
    return provider.generate_summary(explanation_prompt.SYSTEM_PROMPT, user_prompt)
