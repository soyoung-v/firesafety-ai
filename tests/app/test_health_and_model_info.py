"""GET /health, GET /model-info 테스트."""

from __future__ import annotations


def test_health_returns_200_and_status_up(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("UP", "DEGRADED")  # 기존 status 필드 유지
    assert "models" in body


def test_health_reports_all_four_models_loaded(client):
    r = client.get("/health")
    models = r.json()["models"]
    assert set(models.keys()) == {"arcClassifier", "riskClassifier", "anomalyDetector", "currentRegressor"}
    assert all(models.values())  # 전부 로드돼 있어야 함(artifacts/ 실제 존재)


def test_health_does_not_leak_local_paths(client):
    r = client.get("/health")
    body_text = r.text
    assert "/Users/" not in body_text
    assert "artifacts/" not in body_text


def test_model_info_returns_200(client):
    r = client.get("/model-info")
    assert r.status_code == 200


def test_model_info_contains_note_about_synthetic_dataset(client):
    r = client.get("/model-info")
    assert "Synthetic" in r.json()["note"]


def test_model_info_has_all_four_models_with_public_fields(client):
    models = client.get("/model-info").json()["models"]
    for name in ("arcClassifier", "riskClassifier", "anomalyDetector", "currentRegressor"):
        entry = models[name]
        assert entry["loaded"] is True
        assert entry["featureCount"] > 0
        assert entry["windowSize"] == 60


def test_model_info_does_not_leak_local_paths_or_company_info(client):
    r = client.get("/model-info")
    body_text = r.text
    assert "/Users/" not in body_text
    assert ".joblib" not in body_text
    assert "CSTech" not in body_text
