"""LLM 설명 provider 추상화.

Service 코드가 특정 SDK(OpenAI 등) 호출에 직접 결합되지 않도록 최소한의 인터페이스만 둔다.
향후 Claude/Gemini/Local LLM으로 교체하려면 이 인터페이스만 구현하면 된다. LangChain 같은
프레임워크는 이번 단계의 단순한 요구사항에는 과하다고 판단해 쓰지 않는다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ProviderError(RuntimeError):
    """provider 호출 실패(공통)."""


class ProviderTimeoutError(ProviderError):
    """provider 응답 지연."""


class ProviderResponseError(ProviderError):
    """provider가 구조화된 형식(analysisSummary)대로 응답하지 않음."""


class ExplanationProvider(ABC):
    # system/user prompt를 받아 analysisSummary 문자열 1개를 생성한다. 실패 시 ProviderError 계열 예외를 던진다.
    @abstractmethod
    def generate_summary(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError
