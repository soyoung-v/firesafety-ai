"""LLM 설명 기능 환경설정. `.env`는 여기서 한 번만 로드한다.

비밀값(OPENAI_API_KEY)은 프로세스 환경변수로만 전달받는다 - 코드/로그/응답 어디에도 값 자체를 남기지 않는다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class LlmSettings:
    enabled: bool
    provider: str
    openai_api_key: str | None
    openai_model: str
    timeout_seconds: float


# .env.example처럼 키만 있고 값이 비어 있는 경우(빈 문자열)도 "미설정"으로 취급한다
def _env(name: str, default: str) -> str:
    value = (os.getenv(name) or "").strip()
    return value if value else default


# 환경변수 -> LlmSettings. 값이 없으면 안전한 기본값(비활성화)으로 떨어진다.
def load_llm_settings() -> LlmSettings:
    return LlmSettings(
        enabled=_env("LLM_EXPLANATION_ENABLED", "false").lower() == "true",
        provider=_env("LLM_PROVIDER", "openai").lower(),
        openai_api_key=(os.getenv("OPENAI_API_KEY") or "").strip() or None,
        openai_model=_env("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        timeout_seconds=float(_env("LLM_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))),
    )
