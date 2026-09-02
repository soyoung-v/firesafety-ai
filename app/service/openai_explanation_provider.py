"""OpenAI Chat Completions 기반 ExplanationProvider 구현.

Structured Output(`response_format`에 Pydantic 모델)을 써서 analysisSummary가 항상 문자열로
검증되도록 한다 - provider 응답을 자유문자열로 무조건 신뢰하지 않는다.
"""

from __future__ import annotations

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI
from pydantic import BaseModel

from .explanation_provider import ExplanationProvider, ProviderError, ProviderResponseError, ProviderTimeoutError


class _SummaryOutput(BaseModel):
    analysisSummary: str


class OpenAiExplanationProvider(ExplanationProvider):
    def __init__(self, api_key: str, model: str, timeout_seconds: float) -> None:
        self._client = OpenAI(api_key=api_key, timeout=timeout_seconds)
        self._model = model

    def generate_summary(self, system_prompt: str, user_prompt: str) -> str:
        try:
            completion = self._client.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=_SummaryOutput,
            )
        except APITimeoutError as exc:
            raise ProviderTimeoutError("OpenAI 응답 지연") from exc
        except (APIConnectionError, APIError) as exc:
            raise ProviderError("OpenAI 호출 실패") from exc

        message = completion.choices[0].message
        if message.refusal:
            raise ProviderResponseError("OpenAI가 응답을 거부했습니다")
        if message.parsed is None or not message.parsed.analysisSummary.strip():
            raise ProviderResponseError("OpenAI 응답이 구조화된 형식이 아닙니다")
        return message.parsed.analysisSummary
