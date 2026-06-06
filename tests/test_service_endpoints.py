from fastapi.testclient import TestClient

import app.main as main
from app.model_loader import ModelLoadMetadata, ModelServiceState


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
    monkeypatch.setenv("MODEL_NAME", "passwords")

    def fake_reload_model(expected_model_version=None):
        assert expected_model_version is None
        return ModelLoadMetadata(
            model_name="passwords",
            model_alias="prod",
            requested_model_version=None,
            loaded_model_version="11",
            model_uri="models:/passwords@prod",
            reloaded_at="2026-06-05T00:00:00+00:00",
        )

    monkeypatch.setattr(main, "reload_model", fake_reload_model)

    response = client.post(
        "/reload_model",
        headers={"X-Service-Token": "test-secret"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "model_reloaded",
        "model_name": "passwords",
        "model_alias": "prod",
        "requested_model_version": None,
        "loaded_model_version": "11",
        "model_uri": "models:/passwords@prod",
        "reloaded_at": "2026-06-05T00:00:00+00:00",
    }


def test_reload_model_with_expected_version(monkeypatch):
    monkeypatch.setenv("SERVICE_RELOAD_SECRET", "test-secret")
    monkeypatch.setenv("MODEL_NAME", "passwords")

    def fake_reload_model(expected_model_version=None):
        assert expected_model_version == "12"
        return ModelLoadMetadata(
            model_name="passwords",
            model_alias="prod",
            requested_model_version="12",
            loaded_model_version="12",
            model_uri="models:/passwords/12",
            reloaded_at="2026-06-05T00:00:00+00:00",
        )

    monkeypatch.setattr(main, "reload_model", fake_reload_model)

    response = client.post(
        "/reload_model",
        headers={"X-Service-Token": "test-secret"},
        json={
            "model_name": "passwords",
            "model_alias": "prod",
            "expected_model_version": "12",
        },
    )

    assert response.status_code == 200
    assert response.json()["requested_model_version"] == "12"
    assert response.json()["loaded_model_version"] == "12"
    assert response.json()["model_uri"] == "models:/passwords/12"


def test_reload_model_without_secret_header(monkeypatch):
    monkeypatch.setenv("SERVICE_RELOAD_SECRET", "test-secret")

    response = client.post("/reload_model")

    assert response.status_code == 401


def test_health_includes_model_diagnostics(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_model_state",
        lambda: ModelServiceState(
            model_loaded=True,
            model_name="passwords",
            model_alias="prod",
            loaded_version="11",
            model_uri="models:/passwords@prod",
            loaded_at="2026-06-05T00:00:00+00:00",
            last_reload_status="success",
            last_reload_error=None,
        ),
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_name": "passwords",
        "model_alias": "prod",
        "loaded_version": "11",
        "model_uri": "models:/passwords@prod",
        "loaded_at": "2026-06-05T00:00:00+00:00",
        "last_reload_status": "success",
        "last_reload_error": None,
    }


def test_model_state_matches_health_diagnostics(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_model_state",
        lambda: ModelServiceState(
            model_loaded=True,
            model_name="passwords",
            model_alias="prod",
            loaded_version="12",
            model_uri="models:/passwords/12",
            loaded_at="2026-06-05T00:00:00+00:00",
            last_reload_status="success",
            last_reload_error=None,
        ),
    )

    response = client.get("/model_state")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["loaded_version"] == "12"
    assert response.json()["last_reload_status"] == "success"


def test_model_status_returns_model_diagnostics(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_model_state",
        lambda: ModelServiceState(
            model_loaded=True,
            model_name="passwords",
            model_alias="prod",
            loaded_version="13",
            model_uri="models:/passwords/13",
            loaded_at="2026-06-06T00:00:00+00:00",
            last_reload_status="failed",
            last_reload_error="MLflow model version is not ready",
        ),
    )

    response = client.get("/model_status")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_name": "passwords",
        "model_alias": "prod",
        "loaded_version": "13",
        "model_uri": "models:/passwords/13",
        "loaded_at": "2026-06-06T00:00:00+00:00",
        "last_reload_status": "failed",
        "last_reload_error": "MLflow model version is not ready",
    }
