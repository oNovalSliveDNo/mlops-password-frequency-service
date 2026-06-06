import logging
import time

import pytest
from fastapi.testclient import TestClient

import app.main as main
import app.model_loader as model_loader
from app.model_loader import ModelLoadMetadata, ModelServiceState


client = TestClient(main.app)


def test_trigger_accepts_pipeline_task(monkeypatch):
    calls = []

    def fake_trigger_training_pipeline(data_url: str):
        calls.append(data_url)

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
    assert response.json() == {
        "status": "accepted",
        "pipeline_id": None,
        "message": None,
    }
    assert calls == ["https://example.com/data.csv"]


def test_trigger_handler_queues_task_without_waiting(monkeypatch):
    called = False

    def slow_trigger_training_pipeline(data_url: str):
        nonlocal called
        called = True
        time.sleep(1)

    monkeypatch.setattr(
        main,
        "trigger_training_pipeline",
        slow_trigger_training_pipeline,
    )

    background_tasks = main.BackgroundTasks()
    response = main.Response()
    start = time.perf_counter()

    result = main.trigger(
        main.TriggerRequest(data_url="https://example.com/data.csv"),
        background_tasks,
        response,
    )

    assert time.perf_counter() - start < 0.1
    assert response.status_code == 202
    assert result.status == "accepted"
    assert called is False


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
        "instance_id": "",
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
        "instance_id": "",
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
    assert response.json()["instance_id"] == ""
    assert response.json()["loaded_version"] == "12"
    assert response.json()["model_uri"] == "models:/passwords/12"
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
        "instance_id": "",
        "model_loaded": True,
        "model_name": "passwords",
        "model_alias": "prod",
        "loaded_version": "13",
        "model_uri": "models:/passwords/13",
        "loaded_at": "2026-06-06T00:00:00+00:00",
        "last_reload_status": "failed",
        "last_reload_error": "MLflow model version is not ready",
    }


def test_predict_returns_serving_metadata_headers(monkeypatch):
    monkeypatch.setattr(main, "predict_passwords", lambda passwords: [1.0])
    monkeypatch.setattr(
        main,
        "get_model_state",
        lambda: ModelServiceState(
            instance_id="instance-a",
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

    response = client.post("/predict", json={"Password": ["password"]})

    assert response.status_code == 200
    assert response.headers["X-Instance-ID"] == "instance-a"
    assert response.headers["X-Model-Version"] == "12"
    assert response.json() == {"Times": [1.0]}


def test_predict_error_detail_does_not_expose_secret(monkeypatch, caplog):
    secret = "predict-secret"
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", secret)

    def fake_predict_passwords(passwords):
        raise RuntimeError(f"failed to load https://user:{secret}@mlflow.example")

    monkeypatch.setattr(main, "predict_passwords", fake_predict_passwords)

    with caplog.at_level(logging.ERROR):
        response = client.post("/predict", json={"Password": ["password"]})

    assert response.status_code == 503
    assert response.json()["detail"] == "Model is currently unavailable"
    assert secret not in response.text
    assert secret not in caplog.text
    assert "[REDACTED]" in caplog.text


def test_trigger_background_error_does_not_expose_secret(monkeypatch, caplog):
    secret = "gitlab-trigger-secret"
    monkeypatch.setenv("GITLAB_TRIGGER_TOKEN", secret)

    def fake_trigger_training_pipeline(data_url: str):
        raise RuntimeError(f"bad trigger token {secret}")

    monkeypatch.setattr(
        main, "trigger_training_pipeline", fake_trigger_training_pipeline
    )

    with caplog.at_level(logging.ERROR):
        response = client.post(
            "/trigger",
            json={"data_url": "https://example.com/data.csv"},
        )

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
    assert secret not in response.text
    assert secret not in caplog.text
    assert "[REDACTED]" in caplog.text


def test_reload_model_error_detail_does_not_expose_secret(monkeypatch, caplog):
    secret = "reload-secret"
    monkeypatch.setenv("SERVICE_RELOAD_SECRET", secret)
    monkeypatch.setenv("MODEL_NAME", "passwords")

    def fake_reload_model(expected_model_version=None):
        raise RuntimeError(f"reload failed with token {secret}")

    monkeypatch.setattr(main, "reload_model", fake_reload_model)

    with caplog.at_level(logging.ERROR):
        response = client.post(
            "/reload_model",
            headers={"X-Service-Token": secret},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model reload failed"
    assert secret not in response.text
    assert secret not in caplog.text
    assert "[REDACTED]" in caplog.text


def test_health_last_reload_error_uses_safe_category(monkeypatch):
    secret = "mlflow-password"
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", secret)
    model_loader.set_model_for_tests(None)

    def fake_load_model_from_mlflow(expected_model_version=None):
        raise RuntimeError(f"failed to load https://user:{secret}@mlflow.example")

    monkeypatch.setattr(
        model_loader,
        "load_model_from_mlflow",
        fake_load_model_from_mlflow,
    )

    with pytest.raises(RuntimeError):
        model_loader.get_model()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["last_reload_status"] == "failed"
    assert response.json()["last_reload_error"] == "model_reload_failed:RuntimeError"
    assert secret not in response.text


def test_health_is_liveness_when_model_not_loaded(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_model_state",
        lambda: ModelServiceState(
            model_loaded=False,
            model_name=None,
            model_alias=None,
            loaded_version=None,
            model_uri=None,
            loaded_at=None,
            last_reload_status="not_loaded",
            last_reload_error=None,
        ),
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is False
    assert response.json()["last_reload_status"] == "not_loaded"


def test_ready_returns_503_when_model_not_loaded(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_model_state",
        lambda: ModelServiceState(
            model_loaded=False,
            model_name=None,
            model_alias=None,
            loaded_version=None,
            model_uri=None,
            loaded_at=None,
            last_reload_status="not_loaded",
            last_reload_error=None,
        ),
    )

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["model_loaded"] is False
    assert response.json()["last_reload_status"] == "not_loaded"


def test_ready_returns_200_when_model_loaded(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_model_state",
        lambda: ModelServiceState(
            model_loaded=True,
            model_name="passwords",
            model_alias="prod",
            loaded_version="14",
            model_uri="models:/passwords/14",
            loaded_at="2026-06-06T00:00:00+00:00",
            last_reload_status="success",
            last_reload_error=None,
        ),
    )

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "instance_id": "",
        "model_loaded": True,
        "model_name": "passwords",
        "model_alias": "prod",
        "loaded_version": "14",
        "model_uri": "models:/passwords/14",
        "loaded_at": "2026-06-06T00:00:00+00:00",
        "last_reload_status": "success",
        "last_reload_error": None,
    }


def test_ready_returns_503_after_last_load_failure(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_model_state",
        lambda: ModelServiceState(
            model_loaded=False,
            model_name=None,
            model_alias=None,
            loaded_version=None,
            model_uri=None,
            loaded_at=None,
            last_reload_status="failed",
            last_reload_error="model_reload_failed:RuntimeError",
        ),
    )

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["model_loaded"] is False
    assert response.json()["last_reload_status"] == "failed"
    assert response.json()["last_reload_error"] == "model_reload_failed:RuntimeError"
