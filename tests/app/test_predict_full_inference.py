"""확장(context 포함) 요청으로 4개 모델 전체 추론이 실행되는지 검증하는 integration test."""

from __future__ import annotations


def test_extended_request_returns_200(client, extended_request):
    r = client.post("/predict", json=extended_request)
    assert r.status_code == 200


def test_extended_request_computes_all_five_outputs(client, extended_request):
    result = client.post("/predict", json=extended_request).json()["results"][0]
    assert result["pred"] in (0, 1)
    assert 0.0 <= result["proba"] <= 1.0
    assert result["riskLevel"] in ("NORMAL", "WARNING", "DANGER")
    assert 0.0 <= result["riskScore"] <= 1.0
    assert isinstance(result["anomaly"], bool)
    assert 0.0 <= result["anomalyScore"] <= 1.0
    assert isinstance(result["predictedCurrent"], float)


def test_extended_request_risk_level_consistent_with_normal_pattern(client, extended_request):
    # 정상 패턴에 가까운 입력이므로 riskLevel=NORMAL, riskScore가 낮아야 한다
    result = client.post("/predict", json=extended_request).json()["results"][0]
    assert result["riskLevel"] == "NORMAL"
    assert result["riskScore"] < 0.5


def test_multi_circuit_request_processes_each_independently(client, circuit_samples_factory, context_samples_factory):
    req = {
        "m_no": "00001",
        "circuits": [
            {"circuit": 1, "samples": circuit_samples_factory(seed=1)},
            {"circuit": 2, "samples": circuit_samples_factory(seed=2)},
            {"circuit": 3, "samples": circuit_samples_factory(seed=3)},
        ],
        "context": {"samples": context_samples_factory()},
    }
    body = client.post("/predict", json=req).json()
    assert len(body["results"]) == 3
    assert [r["circuit"] for r in body["results"]] == [1, 2, 3]
    for result in body["results"]:
        assert result["riskLevel"] is not None


def test_context_length_mismatch_skips_risk_and_anomaly_without_error(client, circuit_samples_factory, context_samples_factory):
    # context 샘플 수가 circuit 샘플 수와 다르면 Risk/Anomaly는 계산하지 않되 요청 자체는 성공해야 한다
    req = {
        "m_no": "00001",
        "circuits": [{"circuit": 1, "samples": circuit_samples_factory(n=60)}],
        "context": {"samples": context_samples_factory(n=45)},
    }
    r = client.post("/predict", json=req)
    assert r.status_code == 200
    result = r.json()["results"][0]
    assert result["riskLevel"] is None
    assert result["anomaly"] is None
    assert result["pred"] in (0, 1)  # ARC는 여전히 계산됨


def test_context_with_missing_required_field_skips_risk_and_anomaly(client, circuit_samples_factory, context_samples_factory):
    samples = context_samples_factory(n=60)
    for s in samples:
        s["temperature"] = None  # 필수 3개 필드 중 하나가 비어있는 경우
    req = {
        "m_no": "00001",
        "circuits": [{"circuit": 1, "samples": circuit_samples_factory(n=60)}],
        "context": {"samples": samples},
    }
    r = client.post("/predict", json=req)
    assert r.status_code == 200
    result = r.json()["results"][0]
    assert result["riskLevel"] is None
    assert result["anomaly"] is None
