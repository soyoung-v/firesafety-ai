"""기존 Spring Boot 요청 형태(legacy) 호환성 테스트 - 신규 필드 추가로 legacy 요청이 깨지면 안 된다."""

from __future__ import annotations


def test_legacy_request_returns_200(client, legacy_request):
    r = client.post("/predict", json=legacy_request)
    assert r.status_code == 200


def test_legacy_request_preserves_m_no(client, legacy_request):
    body = client.post("/predict", json=legacy_request).json()
    assert body["m_no"] == legacy_request["m_no"]


def test_legacy_request_preserves_circuit_number(client, legacy_request):
    body = client.post("/predict", json=legacy_request).json()
    assert body["results"][0]["circuit"] == 1


def test_legacy_request_has_pred_and_proba(client, legacy_request):
    result = client.post("/predict", json=legacy_request).json()["results"][0]
    assert result["pred"] in (0, 1)
    assert 0.0 <= result["proba"] <= 1.0


def test_legacy_request_preserves_threshold(client, legacy_request):
    body = client.post("/predict", json=legacy_request).json()
    assert body["threshold"] == 0.5


def test_legacy_request_preserves_n_samples(client, legacy_request):
    result = client.post("/predict", json=legacy_request).json()["results"][0]
    assert result["n_samples"] == 60


def test_legacy_request_without_context_has_null_risk_and_anomaly_fields(client, legacy_request):
    result = client.post("/predict", json=legacy_request).json()["results"][0]
    assert result["riskLevel"] is None
    assert result["riskScore"] is None
    assert result["anomaly"] is None
    assert result["anomalyScore"] is None


def test_legacy_request_still_computes_predicted_current(client, legacy_request):
    # predictedCurrent는 current 이력만 있으면 계산 가능하므로 legacy 요청에서도 값이 나와야 한다
    result = client.post("/predict", json=legacy_request).json()["results"][0]
    assert result["predictedCurrent"] is not None
    assert isinstance(result["predictedCurrent"], float)


def test_legacy_request_response_field_names_unchanged(client, legacy_request):
    result = client.post("/predict", json=legacy_request).json()["results"][0]
    for field in ("circuit", "proba", "pred", "n_samples", "warning"):
        assert field in result


def test_legacy_request_below_recommended_sample_count_gets_warning(client, circuit_samples_factory):
    req = {"m_no": "00001", "circuits": [{"circuit": 1, "samples": circuit_samples_factory(n=35)}]}
    result = client.post("/predict", json=req).json()["results"][0]
    assert result["warning"] is not None
    assert result["n_samples"] == 35


def test_legacy_request_with_recommended_sample_count_has_no_warning(client, legacy_request):
    result = client.post("/predict", json=legacy_request).json()["results"][0]
    assert result["warning"] is None
