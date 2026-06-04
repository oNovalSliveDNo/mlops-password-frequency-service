import logging

import pytest
import requests

from training.run_pipeline import call_reload_model_endpoint


class FakeResponse:
    def __init__(self, status_code=200, json_body=None, json_error=False):
        self.status_code = status_code
        self._json_body = json_body
        self._json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        if self._json_error:
            raise ValueError("not json")
        return self._json_body


def test_call_reload_model_endpoint_skips_locally_without_url(monkeypatch, caplog):
    monkeypatch.delenv("SERVICE_RELOAD_URL", raising=False)
    monkeypatch.delenv("CI", raising=False)

    with caplog.at_level(logging.WARNING):
        result = call_reload_model_endpoint()

    assert result == {
        "status": "skipped",
        "reason": "SERVICE_RELOAD_URL is not configured",
    }
    assert "SERVICE_RELOAD_URL is not configured" in caplog.text


def test_call_reload_model_endpoint_requires_url_in_ci(monkeypatch):
    monkeypatch.delenv("SERVICE_RELOAD_URL", raising=False)
    monkeypatch.setenv("CI", "true")

    with pytest.raises(RuntimeError, match="SERVICE_RELOAD_URL is required in CI"):
        call_reload_model_endpoint()


def test_call_reload_model_endpoint_posts_with_secret(monkeypatch):
    monkeypatch.setenv("SERVICE_RELOAD_URL", "https://service.example/reload_model")
    monkeypatch.setenv("SERVICE_RELOAD_SECRET", "super-secret")
    calls = []

    def fake_post(url, headers, timeout):
        calls.append({"url": url, "headers": headers, "timeout": timeout})
        return FakeResponse(json_body={"status": "model_reloaded"})

    monkeypatch.setattr("training.run_pipeline.requests.post", fake_post)

    result = call_reload_model_endpoint()

    assert result == {"status": "model_reloaded"}
    assert calls == [
        {
            "url": "https://service.example/reload_model",
            "headers": {"X-Service-Token": "super-secret"},
            "timeout": 10,
        }
    ]


def test_call_reload_model_endpoint_returns_status_for_non_json(monkeypatch):
    monkeypatch.setenv("SERVICE_RELOAD_URL", "https://service.example/reload_model")
    monkeypatch.setenv("SERVICE_RELOAD_SECRET", "super-secret")

    def fake_post(url, headers, timeout):
        return FakeResponse(status_code=204, json_error=True)

    monkeypatch.setattr("training.run_pipeline.requests.post", fake_post)

    assert call_reload_model_endpoint() == {"status": "success", "http_status": 204}


def test_call_reload_model_endpoint_retries_and_sanitizes_secret(monkeypatch):
    monkeypatch.setenv("SERVICE_RELOAD_URL", "https://service.example/reload_model")
    monkeypatch.setenv("SERVICE_RELOAD_SECRET", "super-secret")
    attempts = 0
    sleeps = []

    def fake_post(url, headers, timeout):
        nonlocal attempts
        attempts += 1
        raise requests.ConnectionError("network failure with super-secret")

    monkeypatch.setattr("training.run_pipeline.requests.post", fake_post)
    monkeypatch.setattr("training.run_pipeline.time.sleep", sleeps.append)

    with pytest.raises(RuntimeError) as exc_info:
        call_reload_model_endpoint()

    error_message = str(exc_info.value)
    assert attempts == 3
    assert sleeps == [2, 2]
    assert "ConnectionError" in error_message
    assert "super-secret" not in error_message
