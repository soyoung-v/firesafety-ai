"""요청 검증/에러 처리 테스트 - 기존 계약 의미(400/422/503)를 유지한다."""

from __future__ import annotations


def test_empty_circuits_returns_400(client):
    req = {"m_no": "00001", "circuits": []}
    r = client.post("/predict", json=req)
    assert r.status_code == 400


def test_insufficient_samples_returns_400(client, circuit_samples_factory):
    req = {"m_no": "00001", "circuits": [{"circuit": 1, "samples": circuit_samples_factory(n=10)}]}
    r = client.post("/predict", json=req)
    assert r.status_code == 400


def test_one_circuit_insufficient_samples_fails_whole_request(client, circuit_samples_factory):
    # 여러 circuit 중 하나라도 샘플 부족이면 전체 요청이 실패해야 한다(partial success 없음)
    req = {
        "m_no": "00001",
        "circuits": [
            {"circuit": 1, "samples": circuit_samples_factory(n=60)},
            {"circuit": 2, "samples": circuit_samples_factory(n=5)},
        ],
    }
    r = client.post("/predict", json=req)
    assert r.status_code == 400


def test_circuit_out_of_range_returns_422(client, circuit_samples_factory):
    req = {"m_no": "00001", "circuits": [{"circuit": 11, "samples": circuit_samples_factory()}]}
    r = client.post("/predict", json=req)
    assert r.status_code == 422


def test_circuit_zero_returns_422(client, circuit_samples_factory):
    req = {"m_no": "00001", "circuits": [{"circuit": 0, "samples": circuit_samples_factory()}]}
    r = client.post("/predict", json=req)
    assert r.status_code == 422


def test_negative_current_returns_422(client, circuit_samples_factory):
    samples = circuit_samples_factory()
    samples[0]["am"] = -1.0
    req = {"m_no": "00001", "circuits": [{"circuit": 1, "samples": samples}]}
    r = client.post("/predict", json=req)
    assert r.status_code == 422


def test_negative_arc_count_returns_422(client, circuit_samples_factory):
    samples = circuit_samples_factory()
    samples[0]["count"] = -1
    req = {"m_no": "00001", "circuits": [{"circuit": 1, "samples": samples}]}
    r = client.post("/predict", json=req)
    assert r.status_code == 422


def test_missing_required_field_returns_422(client, circuit_samples_factory):
    req = {"circuits": [{"circuit": 1, "samples": circuit_samples_factory()}]}  # m_no 누락
    r = client.post("/predict", json=req)
    assert r.status_code == 422


def test_nan_current_rejected(client, circuit_samples_factory):
    # 표준 json.dumps는 NaN을 직렬화하지 못하므로(json= 헬퍼 사용 불가) raw body로 직접 보낸다 -
    # FastAPI/Pydantic은 비표준 NaN 리터럴이 포함된 body도 파싱은 하되 allow_inf_nan=False로 거절해야 한다
    samples = circuit_samples_factory()
    body = '{"m_no": "00001", "circuits": [{"circuit": 1, "samples": [{"am": NaN, "count": 0}' + "".join(
        f', {{"am": {s["am"]}, "count": {s["count"]}}}' for s in samples[1:]
    ) + "]}]}"
    r = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_infinite_current_rejected(client, circuit_samples_factory):
    samples = circuit_samples_factory()
    body = '{"m_no": "00001", "circuits": [{"circuit": 1, "samples": [{"am": Infinity, "count": 0}' + "".join(
        f', {{"am": {s["am"]}, "count": {s["count"]}}}' for s in samples[1:]
    ) + "]}]}"
    r = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_error_response_does_not_leak_local_paths(client, circuit_samples_factory):
    req = {"m_no": "00001", "circuits": [{"circuit": 1, "samples": circuit_samples_factory(n=5)}]}
    r = client.post("/predict", json=req)
    assert "/Users/" not in r.text
    assert "Traceback" not in r.text
