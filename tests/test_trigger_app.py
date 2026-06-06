import subprocess
import sys
import time

import pytest
import requests
from fastapi.testclient import TestClient

import trigger_app.main as trigger_main


client = TestClient(trigger_main.app)


class FakeResponse:
    def __init__(self, payload, status_error=None):
        self._payload = payload
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error is not None:
            raise self._status_error

    def json(self):
        return self._payload


def _set_required_gitlab_env(monkeypatch):
    monkeypatch.setenv("GITLAB_URL", "https://gitlab.example/")
    monkeypatch.setenv("GITLAB_PROJECT_ID", "group/project")
    monkeypatch.setenv("GITLAB_TRIGGER_TOKEN", "trigger-token")
    monkeypatch.delenv("GITLAB_TOKEN", raising=False)


def test_trigger_app_import_does_not_import_ml_serving_modules():
    script = """
import importlib
import sys
importlib.import_module('trigger_app.main')
for module_name in ('app.main', 'app.model_loader', 'mlflow', 'sklearn', 'pandas'):
    assert module_name not in sys.modules, module_name
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_serving_app_import_does_not_import_model_loader():
    script = """
import importlib
import sys
importlib.import_module('app.main')
for module_name in ('app.model_loader', 'mlflow', 'sklearn', 'pandas'):
    assert module_name not in sys.modules, module_name
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_trigger_app_health_is_lightweight():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_trigger_app_accepts_pipeline_task(monkeypatch):
    calls = []

    def fake_trigger_training_pipeline(data_url: str):
        calls.append(data_url)

    monkeypatch.setattr(
        trigger_main,
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


def test_trigger_app_handler_queues_task_without_waiting(monkeypatch):
    called = False

    def slow_trigger_training_pipeline(data_url: str):
        nonlocal called
        called = True
        time.sleep(1)

    monkeypatch.setattr(
        trigger_main,
        "trigger_training_pipeline",
        slow_trigger_training_pipeline,
    )

    background_tasks = trigger_main.BackgroundTasks()
    response = trigger_main.Response()
    start = time.perf_counter()

    result = trigger_main.trigger(
        trigger_main.TriggerRequest(data_url="https://example.com/data.csv"),
        background_tasks,
        response,
    )

    assert time.perf_counter() - start < 0.1
    assert response.status_code == 202
    assert result.status == "accepted"
    assert called is False


def test_trigger_app_invalid_payload():
    response = client.post("/trigger", json={"data_url": ""})

    assert response.status_code == 422


def test_trigger_app_pipeline_uses_trigger_endpoint_without_project_token(monkeypatch):
    _set_required_gitlab_env(monkeypatch)
    monkeypatch.setenv("GITLAB_REF", "feature/ref")
    calls = []

    def fake_post(url, data, timeout):
        calls.append({"url": url, "data": data, "timeout": timeout})
        return FakeResponse(
            {
                "id": 123,
                "web_url": "https://gitlab.example/group/project/-/pipelines/123",
            }
        )

    monkeypatch.setattr(trigger_main.requests, "post", fake_post)

    result = trigger_main.trigger_training_pipeline("https://example.com/data.csv")

    assert result == {
        "pipeline_id": 123,
        "web_url": "https://gitlab.example/group/project/-/pipelines/123",
    }
    assert calls == [
        {
            "url": "https://gitlab.example/api/v4/projects/group%2Fproject/trigger/pipeline",
            "data": {
                "token": "trigger-token",
                "ref": "feature/ref",
                "variables[DATA_URL]": "https://example.com/data.csv",
            },
            "timeout": 5,
        }
    ]


@pytest.mark.parametrize(
    "missing_env",
    ["GITLAB_URL", "GITLAB_PROJECT_ID", "GITLAB_TRIGGER_TOKEN"],
)
def test_trigger_app_pipeline_requires_minimal_trigger_env(monkeypatch, missing_env):
    _set_required_gitlab_env(monkeypatch)
    monkeypatch.delenv(missing_env, raising=False)

    with pytest.raises(
        RuntimeError, match=f"Environment variable {missing_env} is required"
    ):
        trigger_main.trigger_training_pipeline("https://example.com/data.csv")


def test_trigger_app_pipeline_propagates_gitlab_http_errors(monkeypatch):
    _set_required_gitlab_env(monkeypatch)
    error = requests.HTTPError("403 Client Error")

    def fake_post(url, data, timeout):
        return FakeResponse({}, status_error=error)

    monkeypatch.setattr(trigger_main.requests, "post", fake_post)

    with pytest.raises(requests.HTTPError, match="403 Client Error"):
        trigger_main.trigger_training_pipeline("https://example.com/data.csv")


def test_trigger_app_background_error_does_not_expose_secret(monkeypatch, caplog):
    secret = "gitlab-trigger-secret"
    monkeypatch.setenv("GITLAB_TRIGGER_TOKEN", secret)

    def fake_trigger_training_pipeline(data_url: str):
        raise RuntimeError(f"bad trigger token {secret}")

    monkeypatch.setattr(
        trigger_main, "trigger_training_pipeline", fake_trigger_training_pipeline
    )

    with caplog.at_level("ERROR"):
        response = client.post(
            "/trigger",
            json={"data_url": "https://example.com/data.csv"},
        )

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
    assert secret not in response.text
    assert secret not in caplog.text
    assert "[REDACTED]" in caplog.text
