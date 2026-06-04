import logging
from dataclasses import dataclass
import pytest
import requests

from training.run_pipeline import call_reload_model_endpoint


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: list[str]
    n_rows: int
    columns: list[str]


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


def test_run_training_pipeline_stops_on_invalid_data(monkeypatch, caplog):
    from training.run_pipeline import run_training_pipeline

    calls = []

    monkeypatch.setenv("DATA_URL", "https://example.com/data.csv")
    monkeypatch.setattr(
        "training.run_pipeline.download_data",
        lambda data_url, output_path: "downloaded.csv",
    )
    monkeypatch.setattr(
        "training.run_pipeline._read_validated_training_dataframe",
        lambda data_path: calls.append("read_validated"),
    )
    monkeypatch.setattr(
        "training.run_pipeline.run_evidently_tests",
        lambda df, output_path: calls.append("evidently"),
    )
    monkeypatch.setattr(
        "training.run_pipeline.validate_data_file",
        lambda input_path, report_path: ValidationResult(False, ["bad data"], 0, []),
    )
    monkeypatch.setattr(
        "training.run_pipeline.train_password_model",
        lambda df: calls.append("train"),
    )
    monkeypatch.setattr(
        "training.run_pipeline.save_model_artifact",
        lambda model, output_path: calls.append("save"),
    )
    monkeypatch.setattr(
        "training.run_pipeline.register_model_in_mlflow",
        lambda model, metrics, validation_report=None: calls.append("register"),
    )
    monkeypatch.setattr(
        "training.run_pipeline.call_reload_model_endpoint",
        lambda: calls.append("reload"),
    )

    with caplog.at_level(logging.ERROR):
        result = run_training_pipeline()

    assert result == {
        "status": "validation_failed",
        "data_path": "downloaded.csv",
        "validation_report": {
            "is_valid": False,
            "errors": ["bad data"],
            "n_rows": 0,
            "columns": [],
        },
        "errors": ["bad data"],
    }
    assert "Data validation failed: bad data" in caplog.text
    assert calls == []


def test_main_allows_validation_failed_result(monkeypatch):
    from training.run_pipeline import main

    monkeypatch.setattr(
        "training.run_pipeline.run_training_pipeline",
        lambda: {"status": "validation_failed", "errors": ["bad data"]},
    )

    main()


def test_run_training_pipeline_does_not_reload_when_registration_fails(monkeypatch):
    from training.run_pipeline import run_training_pipeline

    calls = []

    monkeypatch.setenv("DATA_URL", "https://example.com/data.csv")
    monkeypatch.setattr(
        "training.run_pipeline.download_data",
        lambda data_url, output_path: "downloaded.csv",
    )
    monkeypatch.setattr("training.run_pipeline.read_csv", lambda path: object())
    monkeypatch.setattr(
        "training.run_pipeline.run_evidently_tests",
        lambda df, output_path: {"status": "success"},
    )
    monkeypatch.setattr(
        "training.run_pipeline.validate_data_file",
        lambda input_path, report_path: ValidationResult(
            True, [], 1, ["Password", "Times"]
        ),
    )
    monkeypatch.setattr(
        "training.run_pipeline.validate_password_dataframe",
        lambda df: (True, [], object()),
    )
    monkeypatch.setattr(
        "training.run_pipeline.train_password_model",
        lambda df: ("model", {"n_rows": 1}),
    )
    monkeypatch.setattr(
        "training.run_pipeline.save_model_artifact",
        lambda model, output_path: "artifacts/model.joblib",
    )

    def fail_registration(model, metrics, validation_report=None):
        calls.append("register")
        raise RuntimeError("mlflow is down")

    monkeypatch.setattr(
        "training.run_pipeline.register_model_in_mlflow",
        fail_registration,
    )
    monkeypatch.setattr(
        "training.run_pipeline.call_reload_model_endpoint",
        lambda: calls.append("reload"),
    )

    with pytest.raises(RuntimeError, match="mlflow is down"):
        run_training_pipeline()

    assert calls == ["register"]


def test_run_training_pipeline_reloads_after_prod_registration(monkeypatch):
    from training.run_pipeline import run_training_pipeline

    calls = []

    monkeypatch.setenv("DATA_URL", "https://example.com/data.csv")
    monkeypatch.setattr(
        "training.run_pipeline.download_data",
        lambda data_url, output_path: "downloaded.csv",
    )
    monkeypatch.setattr("training.run_pipeline.read_csv", lambda path: object())
    monkeypatch.setattr(
        "training.run_pipeline.run_evidently_tests",
        lambda df, output_path: {"status": "success"},
    )
    monkeypatch.setattr(
        "training.run_pipeline.validate_data_file",
        lambda input_path, report_path: ValidationResult(
            True, [], 1, ["Password", "Times"]
        ),
    )
    monkeypatch.setattr(
        "training.run_pipeline.validate_password_dataframe",
        lambda df: (True, [], object()),
    )
    monkeypatch.setattr(
        "training.run_pipeline.train_password_model",
        lambda df: ("model", {"n_rows": 1}),
    )
    monkeypatch.setattr(
        "training.run_pipeline.save_model_artifact",
        lambda model, output_path: "artifacts/model.joblib",
    )

    def register(model, metrics, validation_report=None):
        calls.append(("register", validation_report))
        return {"model_name": "passwords", "model_alias": "prod", "model_version": "1"}

    monkeypatch.setattr("training.run_pipeline.register_model_in_mlflow", register)
    monkeypatch.setattr(
        "training.run_pipeline.call_reload_model_endpoint",
        lambda: calls.append("reload") or {"status": "model_reloaded"},
    )

    result = run_training_pipeline()

    assert calls == [
        (
            "register",
            {
                "is_valid": True,
                "errors": [],
                "n_rows": 1,
                "columns": ["Password", "Times"],
            },
        ),
        "reload",
    ]
    assert result["reload"] == {"status": "model_reloaded"}


def test_run_training_pipeline_requires_reload_url_in_ci_after_registration(
    monkeypatch,
):
    from training.run_pipeline import run_training_pipeline

    monkeypatch.setenv("DATA_URL", "https://example.com/data.csv")
    monkeypatch.setenv("CI", "true")
    monkeypatch.delenv("SERVICE_RELOAD_URL", raising=False)
    monkeypatch.setattr(
        "training.run_pipeline.download_data",
        lambda data_url, output_path: "downloaded.csv",
    )
    monkeypatch.setattr("training.run_pipeline.read_csv", lambda path: object())
    monkeypatch.setattr(
        "training.run_pipeline.run_evidently_tests",
        lambda df, output_path: {"status": "success"},
    )
    monkeypatch.setattr(
        "training.run_pipeline.validate_data_file",
        lambda input_path, report_path: ValidationResult(
            True, [], 1, ["Password", "Times"]
        ),
    )
    monkeypatch.setattr(
        "training.run_pipeline.validate_password_dataframe",
        lambda df: (True, [], object()),
    )
    monkeypatch.setattr(
        "training.run_pipeline.train_password_model",
        lambda df: ("model", {"n_rows": 1}),
    )
    monkeypatch.setattr(
        "training.run_pipeline.save_model_artifact",
        lambda model, output_path: "artifacts/model.joblib",
    )
    monkeypatch.setattr(
        "training.run_pipeline.register_model_in_mlflow",
        lambda model, metrics, validation_report=None: {
            "model_name": "passwords",
            "model_alias": "prod",
            "model_version": "1",
        },
    )

    with pytest.raises(RuntimeError, match="SERVICE_RELOAD_URL is required in CI"):
        run_training_pipeline()
