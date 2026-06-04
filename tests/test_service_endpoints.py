from fastapi.testclient import TestClient

import app.main as main


client = TestClient(main.app)


def test_trigger_starts_pipeline(monkeypatch):
    def fake_trigger_training_pipeline(data_url: str):
        assert data_url == "https://example.com/data.csv"
        return {
            "pipeline_id": 123,
            "web_url": "https://gitlab/pipeline/123",
        }

    monkeypatch.setattr(
        main,
        "trigger_training_pipeline",
        fake_trigger_training_pipeline,
    )

    response = client.post(
        "/trigger",
        json={"data_url": "https://example.com/data.csv"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "started"
    assert body["pipeline_id"] == 123


def test_trigger_invalid_payload():
    response = client.post("/trigger", json={"data_url": ""})

    assert response.status_code == 422


def test_reload_model_with_valid_secret(monkeypatch):
    monkeypatch.setenv("SERVICE_RELOAD_SECRET", "test-secret")

    def fake_reload_model():
        return None

    monkeypatch.setattr(main, "reload_model", fake_reload_model)

    response = client.post(
        "/reload_model",
        headers={"X-Service-Token": "test-secret"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "model_reloaded"


def test_reload_model_without_secret_header(monkeypatch):
    monkeypatch.setenv("SERVICE_RELOAD_SECRET", "test-secret")

    response = client.post("/reload_model")

    assert response.status_code == 401
