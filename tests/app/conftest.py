"""tests/app 공용 fixture. 실제 artifacts/의 학습된 모델을 로드해 FastAPI 앱을 띄운다."""

from __future__ import annotations

import random

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def circuit_samples_factory():
    # 재현 가능한 정상 패턴 회로 샘플 n개 생성 (am 5A 부근 + 약한 노이즈)
    def _make(n: int = 60, seed: int = 1) -> list[dict]:
        rng = random.Random(seed)
        return [{"am": round(5.0 + rng.gauss(0, 0.3), 3), "count": 0} for _ in range(n)]

    return _make


@pytest.fixture
def context_samples_factory():
    # 재현 가능한 정상 패턴 context 샘플 n개 생성 (Dataset Generator NORMAL baseline과 유사한 분포)
    def _make(n: int = 60, seed: int = 2) -> list[dict]:
        rng = random.Random(seed)
        return [
            {
                "voltage": round(224 + rng.gauss(0, 3), 2),
                "leakage_current": round(2.0 + rng.gauss(0, 0.5), 3),
                "temperature": round(27.0 + rng.gauss(0, 1.5), 3),
                "humidity": round(45 + rng.gauss(0, 4), 2),
                "fire_raw": 500,
                "gas_raw": 500,
                "door_open": False,
                "total_current": round(10.0 + rng.gauss(0, 1.0), 3),
                "total_power": 2000,
            }
            for _ in range(n)
        ]

    return _make


@pytest.fixture
def legacy_request(circuit_samples_factory) -> dict:
    return {"m_no": "00001", "circuits": [{"circuit": 1, "samples": circuit_samples_factory()}]}


@pytest.fixture
def extended_request(circuit_samples_factory, context_samples_factory) -> dict:
    return {
        "m_no": "00001",
        "circuits": [{"circuit": 1, "samples": circuit_samples_factory()}],
        "context": {"samples": context_samples_factory()},
    }
