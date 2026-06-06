import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from training.run_pipeline import (
    _RELOAD_RETRY_DELAY_SECONDS,
    _RELOAD_TIMEOUT_SECONDS,
)
from training.run_pipeline import (
    call_reload_model_endpoint,
    pd as pipeline_pd,
    verify_serving_after_reload,
)
from training.validation_thresholds import DEFAULT_SCHEMA_THRESHOLDS


SCHEMA_THRESHOLDS = dict(DEFAULT_SCHEMA_THRESHOLDS)


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: list[str]
    n_rows: int
    columns: list[str]


@dataclass
class ModelQualityValidationResult:
    is_valid: bool
    errors: list[str]
    metrics: dict
    scored_df: object | None = None


class FakeDataFrame:
    def __init__(self, data):
        self._data = {key: list(value) for key, value in data.items()}
        self.columns = list(self._data)
        row_count = len(next(iter(self._data.values()))) if self._data else 0
        self.index = list(range(row_count))

    def __getitem__(self, key):
        if isinstance(key, list):
            return FakeDataFrame({column: self._data[column] for column in key})

        return self._data[key]

    def assign(self, **kwargs):
        data = {key: list(value) for key, value in self._data.items()}
        for key, value in kwargs.items():
            data[key] = list(value)
        return FakeDataFrame(data)

    def copy(self):
        return FakeDataFrame(self._data)

    def equals(self, other):
        return isinstance(other, FakeDataFrame) and self._data == other._data


class FakeTrainingFrame:
    def __getitem__(self, key):
        return self


def _assert_duration_logs(caplog, expected_step_names):
    duration_records = [
        record for record in caplog.records if record.msg == "%s took %.3f sec"
    ]
    actual_step_names = [record.args[0] for record in duration_records]
    assert actual_step_names == expected_step_names
    for record in duration_records:
        assert isinstance(record.args[1], float)
        assert record.args[1] >= 0


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


def test_call_reload_model_endpoint_requires_secret_when_url_is_configured(monkeypatch):
    monkeypatch.setenv("SERVICE_RELOAD_URL", "https://service.example/reload_model")
    monkeypatch.delenv("SERVICE_RELOAD_SECRET", raising=False)
    monkeypatch.setenv("CI", "true")

    def fail_post(*args, **kwargs):
        pytest.fail("reload endpoint must not be called without SERVICE_RELOAD_SECRET")

    monkeypatch.setattr("training.run_pipeline.requests.post", fail_post)

    with pytest.raises(
        RuntimeError,
        match="SERVICE_RELOAD_SECRET is required in CI to call /reload_model",
    ):
        call_reload_model_endpoint()


def test_call_reload_model_endpoint_posts_with_secret(monkeypatch):
    monkeypatch.setenv("SERVICE_RELOAD_URL", "https://service.example/reload_model")
    monkeypatch.setenv("SERVICE_RELOAD_SECRET", "super-secret")
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse(json_body={"status": "model_reloaded"})

    monkeypatch.setattr("training.run_pipeline.requests.post", fake_post)

    result = call_reload_model_endpoint()

    assert result == {"status": "model_reloaded"}
    assert calls == [
        {
            "url": "https://service.example/reload_model",
            "headers": {"X-Service-Token": "super-secret"},
            "json": None,
            "timeout": _RELOAD_TIMEOUT_SECONDS,
        }
    ]


def test_call_reload_model_endpoint_posts_expected_model_version(monkeypatch):
    monkeypatch.setenv("SERVICE_RELOAD_URL", "https://service.example/reload_model")
    monkeypatch.setenv("SERVICE_RELOAD_SECRET", "super-secret")
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse(
            json_body={
                "status": "model_reloaded",
                "requested_model_version": "12",
                "loaded_model_version": "12",
                "model_uri": "models:/passwords/12",
                "reloaded_at": "2026-06-05T00:00:00+00:00",
            }
        )

    monkeypatch.setattr("training.run_pipeline.requests.post", fake_post)

    result = call_reload_model_endpoint(
        {
            "model_name": "passwords",
            "model_alias": "prod",
            "model_version": "12",
        }
    )

    assert result["loaded_model_version"] == "12"
    assert calls == [
        {
            "url": "https://service.example/reload_model",
            "headers": {"X-Service-Token": "super-secret"},
            "json": {
                "model_name": "passwords",
                "model_alias": "prod",
                "expected_model_version": "12",
            },
            "timeout": _RELOAD_TIMEOUT_SECONDS,
        }
    ]


def test_call_reload_model_endpoint_returns_status_for_non_json(monkeypatch):
    monkeypatch.setenv("SERVICE_RELOAD_URL", "https://service.example/reload_model")
    monkeypatch.setenv("SERVICE_RELOAD_SECRET", "super-secret")

    def fake_post(url, headers, json, timeout):
        assert json is None
        return FakeResponse(status_code=204, json_error=True)

    monkeypatch.setattr("training.run_pipeline.requests.post", fake_post)

    assert call_reload_model_endpoint() == {"status": "success", "http_status": 204}


def test_call_reload_model_endpoint_retries_and_sanitizes_secret(monkeypatch):
    monkeypatch.setenv("SERVICE_RELOAD_URL", "https://service.example/reload_model")
    monkeypatch.setenv("SERVICE_RELOAD_SECRET", "super-secret")
    attempts = 0
    sleeps = []

    def fake_post(url, headers, json, timeout):
        nonlocal attempts
        assert json is None
        attempts += 1
        raise requests.ConnectionError("network failure with super-secret")

    monkeypatch.setattr("training.run_pipeline.requests.post", fake_post)
    monkeypatch.setattr("training.run_pipeline.time.sleep", sleeps.append)

    with pytest.raises(RuntimeError) as exc_info:
        call_reload_model_endpoint()

    error_message = str(exc_info.value)
    assert attempts == 3
    assert sleeps == [_RELOAD_RETRY_DELAY_SECONDS, _RELOAD_RETRY_DELAY_SECONDS]
    assert "ConnectionError" in error_message
    assert "super-secret" not in error_message


def test_verify_serving_after_reload_checks_health_and_predict(monkeypatch):
    monkeypatch.setenv("SERVICE_RELOAD_URL", "https://service.example/reload_model")
    calls = []

    training_df = pipeline_pd.DataFrame(
        {
            "Password": ["alpha", "beta"],
            "Times": [9.0, 99.0],
        }
    )

    def fake_request(method, url, timeout, **kwargs):
        calls.append(
            {"method": method, "url": url, "timeout": timeout, "kwargs": kwargs}
        )
        if method == "GET":
            return FakeResponse(
                json_body={
                    "status": "ok",
                    "model_loaded": True,
                    "model_name": "passwords",
                    "model_alias": "prod",
                    "loaded_version": "12",
                    "model_uri": "models:/passwords/12",
                    "loaded_at": "2026-06-05T00:00:00+00:00",
                    "last_reload_status": "success",
                    "last_reload_error": None,
                }
            )

        return FakeResponse(json_body={"Times": [1.0, 2.0]})

    monkeypatch.setattr(
        "training.run_pipeline.requests.request", fake_request, raising=False
    )

    result = verify_serving_after_reload(
        {"model_name": "passwords", "model_alias": "prod", "model_version": "12"},
        {"status": "model_reloaded", "loaded_model_version": "12"},
        training_df,
    )

    assert result["status"] == "verified"
    assert result["expected_model_version"] == "12"
    assert result["loaded_model_version"] == "12"
    assert result["predict"]["url"] == "https://service.example/predict"
    assert result["predict"]["request_count"] == 2
    assert result["predict"]["response_count"] == 2
    assert result["predict"]["checked_rows"] == 2
    assert result["predict"]["max_abs_error"] == 0.0
    assert result["predict"]["rounds"] == 3
    assert result["predict"]["sample_passwords"] == ["alpha", "beta"]
    assert calls == [
        {
            "method": "GET",
            "url": "https://service.example/model_state",
            "timeout": 30,
            "kwargs": {},
        },
        {
            "method": "POST",
            "url": "https://service.example/predict",
            "timeout": 30,
            "kwargs": {"json": {"Password": ["alpha", "beta"]}},
        },
        {
            "method": "GET",
            "url": "https://service.example/model_state",
            "timeout": 30,
            "kwargs": {},
        },
        {
            "method": "POST",
            "url": "https://service.example/predict",
            "timeout": 30,
            "kwargs": {"json": {"Password": ["alpha", "beta"]}},
        },
        {
            "method": "GET",
            "url": "https://service.example/model_state",
            "timeout": 30,
            "kwargs": {},
        },
        {
            "method": "POST",
            "url": "https://service.example/predict",
            "timeout": 30,
            "kwargs": {"json": {"Password": ["alpha", "beta"]}},
        },
    ]


def test_verify_serving_after_reload_rejects_prediction_error(monkeypatch):
    monkeypatch.setenv("SERVICE_RELOAD_URL", "https://service.example/reload_model")
    monkeypatch.setenv("SERVICE_SERVING_VERIFICATION_ROUNDS", "1")
    monkeypatch.setenv("SERVICE_PREDICT_MAX_ABS_ERROR_TOLERANCE", "0.1")
    training_df = pipeline_pd.DataFrame({"Password": ["alpha"], "Times": [9.0]})

    def fake_request(method, url, timeout, **kwargs):
        if method == "GET":
            return FakeResponse(
                json_body={
                    "status": "ok",
                    "model_loaded": True,
                    "loaded_version": "12",
                    "last_reload_status": "success",
                }
            )

        return FakeResponse(json_body={"Times": [9.0]})

    monkeypatch.setattr(
        "training.run_pipeline.requests.request", fake_request, raising=False
    )

    with pytest.raises(RuntimeError, match="max_abs_error tolerance"):
        verify_serving_after_reload(
            {"model_name": "passwords", "model_alias": "prod", "model_version": "12"},
            {"status": "model_reloaded"},
            training_df,
        )


def test_verify_serving_after_reload_checks_configured_replica_state_urls(monkeypatch):
    monkeypatch.setenv("SERVICE_RELOAD_URL", "https://service.example/reload_model")
    monkeypatch.setenv(
        "SERVICE_REPLICA_STATE_URLS",
        "https://replica-a.example/model_state,https://replica-b.example/model_state",
    )
    calls = []
    training_df = pipeline_pd.DataFrame(
        {"Password": ["alpha", "beta"], "Times": [9.0, 99.0]}
    )

    def fake_request(method, url, timeout, **kwargs):
        calls.append(url)
        if method == "GET":
            instance_id = "replica-a" if "replica-a" in url else "replica-b"
            return FakeResponse(
                json_body={
                    "status": "ok",
                    "instance_id": instance_id,
                    "model_loaded": True,
                    "loaded_version": "12",
                    "last_reload_status": "success",
                }
            )

        return FakeResponse(json_body={"Times": [1.0, 2.0]})

    monkeypatch.setattr(
        "training.run_pipeline.requests.request", fake_request, raising=False
    )

    result = verify_serving_after_reload(
        {"model_name": "passwords", "model_alias": "prod", "model_version": "12"},
        {"status": "model_reloaded", "loaded_model_version": "12"},
        training_df,
    )

    assert result["replica_count"] == 2
    assert [item["state"]["instance_id"] for item in result["replicas"]] == [
        "replica-a",
        "replica-b",
    ]
    assert calls[:2] == [
        "https://replica-a.example/model_state",
        "https://replica-b.example/model_state",
    ]


def test_verify_serving_after_reload_rejects_stale_loaded_version(monkeypatch):
    monkeypatch.setenv("SERVICE_HEALTH_URL", "https://service.example/health")
    monkeypatch.setenv("SERVICE_PREDICT_URL", "https://service.example/predict")
    training_df = pipeline_pd.DataFrame({"Password": ["alpha"], "Times": [9.0]})

    def fake_request(method, url, timeout, **kwargs):
        assert method == "GET"
        return FakeResponse(
            json_body={
                "status": "ok",
                "model_loaded": True,
                "loaded_version": "11",
                "last_reload_status": "success",
            }
        )

    monkeypatch.setattr(
        "training.run_pipeline.requests.request", fake_request, raising=False
    )
    with pytest.raises(RuntimeError, match="unexpected model version"):
        verify_serving_after_reload(
            {"model_name": "passwords", "model_alias": "prod", "model_version": "12"},
            {"status": "model_reloaded"},
            training_df,
        )


def test_run_training_pipeline_stops_on_invalid_schema_validation(monkeypatch, caplog):
    from training.run_pipeline import run_training_pipeline

    calls = []

    def fail_if_called(name):
        def inner(*args, **kwargs):
            calls.append(name)
            pytest.fail(f"{name} must not be called after schema validation failure")

        return inner

    monkeypatch.setenv("DATA_URL", "https://example.com/data.csv")
    monkeypatch.setattr(
        "training.run_pipeline.download_data",
        lambda data_url, output_path: "downloaded.csv",
    )
    monkeypatch.setattr(
        "training.run_pipeline._read_validated_training_dataframe",
        fail_if_called("read_validated"),
    )
    schema_metrics = {
        "n_rows": 1,
        "columns": ["Password", "Times"],
        "thresholds": SCHEMA_THRESHOLDS,
    }
    monkeypatch.setattr(
        "training.run_pipeline.validate_data_file",
        lambda input_path, report_path: SimpleNamespace(
            is_valid=False,
            errors=["bad data"],
            n_rows=1,
            columns=["Password", "Times"],
            metrics=schema_metrics,
            thresholds=SCHEMA_THRESHOLDS,
        ),
    )
    monkeypatch.setattr(
        "training.run_pipeline.validate_model_quality_with_prod_model",
        fail_if_called("model_quality"),
    )
    monkeypatch.setattr(
        "training.run_pipeline.run_evidently_tests",
        fail_if_called("evidently"),
    )
    monkeypatch.setattr(
        "training.run_pipeline.train_password_model",
        fail_if_called("train"),
    )
    monkeypatch.setattr(
        "training.run_pipeline.save_model_artifact",
        fail_if_called("save"),
    )
    monkeypatch.setattr(
        "training.run_pipeline.register_model_in_mlflow",
        fail_if_called("register"),
    )
    monkeypatch.setattr(
        "training.run_pipeline.call_reload_model_endpoint",
        fail_if_called("reload"),
    )

    with caplog.at_level(logging.INFO):
        result = run_training_pipeline()

    assert result == {
        "status": "validation_failed",
        "data_path": "downloaded.csv",
        "validation_report": {
            "is_valid": False,
            "errors": ["bad data"],
            "n_rows": 1,
            "columns": ["Password", "Times"],
            "schema_metrics": schema_metrics,
            "thresholds": SCHEMA_THRESHOLDS,
        },
        "errors": ["bad data"],
    }
    assert "data downloaded: downloaded.csv" in caplog.text
    assert "validation failed: bad data" in caplog.text
    assert "VALIDATION_FAILED_NO_MODEL_REGISTERED" in caplog.text
    assert "errors=['bad data']" in caplog.text
    assert "metrics={'n_rows': 1" in caplog.text
    _assert_duration_logs(
        caplog, ["data download", "schema validation", "total pipeline"]
    )
    assert calls == []


def test_run_training_pipeline_exits_on_validation_failed_when_env_enabled(
    monkeypatch,
):
    from training.run_pipeline import run_training_pipeline

    def fail_if_called(*args, **kwargs):
        pytest.fail("training side effects must not run after validation failure")

    monkeypatch.setenv("DATA_URL", "https://example.com/data.csv")
    monkeypatch.setenv("FAIL_CI_ON_VALIDATION_FAILED", "true")
    monkeypatch.setattr(
        "training.run_pipeline.download_data",
        lambda data_url, output_path: "downloaded.csv",
    )
    monkeypatch.setattr(
        "training.run_pipeline.validate_data_file",
        lambda input_path, report_path: SimpleNamespace(
            is_valid=False,
            errors=["bad data"],
            n_rows=1,
            columns=["Password", "Times"],
            metrics={"row_count": 1},
        ),
    )
    monkeypatch.setattr("training.run_pipeline.train_password_model", fail_if_called)
    monkeypatch.setattr(
        "training.run_pipeline.register_model_in_mlflow", fail_if_called
    )
    monkeypatch.setattr(
        "training.run_pipeline.call_reload_model_endpoint", fail_if_called
    )

    with pytest.raises(SystemExit) as exc_info:
        run_training_pipeline()

    assert exc_info.value.code == 2


def test_run_training_pipeline_continues_to_register_on_failed_model_quality(
    tmp_path, monkeypatch, caplog
):
    from training.run_pipeline import run_training_pipeline

    validation_report_path = tmp_path / "validation_report.json"
    training_df = FakeDataFrame(
        {"Password": ["hunter2"], "Times": [10], "source_row": [42]}
    )
    scored_df = training_df.assign(
        target_log=[1.0], prediction=[3.0], prediction_error=[2.0]
    )
    schema_metrics = {
        "row_count": 1,
        "null_passwords": 0,
        "thresholds": SCHEMA_THRESHOLDS,
    }
    model_quality_metrics = {"rmse": 2.0, "mae": 2.0}
    calls = []

    monkeypatch.setenv("DATA_URL", "https://example.com/data.csv")
    monkeypatch.setattr(
        "training.run_pipeline._DEFAULT_VALIDATION_REPORT_PATH",
        str(validation_report_path),
    )
    monkeypatch.setattr(
        "training.run_pipeline.download_data",
        lambda data_url, output_path: "downloaded.csv",
    )
    monkeypatch.setattr(
        "training.run_pipeline.validate_data_file",
        lambda input_path, report_path: SimpleNamespace(
            is_valid=True,
            errors=[],
            n_rows=1,
            columns=["Password", "Times"],
            metrics=schema_metrics,
            cleaned_df=training_df,
            thresholds=SCHEMA_THRESHOLDS,
        ),
    )
    monkeypatch.setattr(
        "training.run_pipeline.validate_model_quality_with_prod_model",
        lambda df: ModelQualityValidationResult(
            False,
            ["rmse too high for token abc123"],
            model_quality_metrics,
            scored_df,
        ),
    )

    def run_evidently(df, output_path):
        calls.append(("evidently", df, output_path))
        return {"status": "success"}

    def train(df):
        calls.append(("train", df.copy()))
        return "model", {"n_rows": len(df.index)}

    def save(model, output_path):
        calls.append(("save", model, output_path))
        return "artifacts/model.joblib"

    def register(model, metrics, validation_report=None, model_alias=None):
        calls.append(("register", model, metrics, validation_report, model_alias))
        return {
            "model_name": "passwords",
            "model_alias": model_alias,
            "model_version": "9",
            "model_uri": "models:/passwords/9",
            "alias_verified": True,
            "verified_model_version": "9",
        }

    monkeypatch.setattr("training.run_pipeline.run_evidently_tests", run_evidently)
    monkeypatch.setattr("training.run_pipeline.train_password_model", train)
    monkeypatch.setattr("training.run_pipeline.save_model_artifact", save)
    monkeypatch.setattr("training.run_pipeline.register_model_in_mlflow", register)

    monkeypatch.setattr(
        "training.run_pipeline.call_reload_model_endpoint",
        lambda registration_result=None: {"status": "skipped"},
    )

    with caplog.at_level(logging.INFO):
        result = run_training_pipeline()

    expected_report = {
        "is_valid": True,
        "errors": [],
        "n_rows": 1,
        "columns": ["Password", "Times"],
        "schema_metrics": schema_metrics,
        "model_quality_is_valid": False,
        "model_quality_errors": ["rmse too high for token abc123"],
        "model_quality_metrics": model_quality_metrics,
        "thresholds": SCHEMA_THRESHOLDS,
    }
    assert result["status"] == "success"
    assert result["validation_report"] == expected_report
    assert result["registration"]["model_version"] == "9"
    written_report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    assert written_report == expected_report
    assert calls[0][0] == "train"
    assert calls[1][0] == "evidently"
    assert calls[2] == ("register", "model", {"n_rows": 1}, expected_report, "prod")
    assert calls[3] == ("save", "model", "artifacts/model.joblib")
    assert (
        "model quality validation failed after data quality checks passed; "
        "continuing to train and register new model version: rmse too high for token abc123"
        in caplog.text
    )
    _assert_duration_logs(
        caplog,
        [
            "data download",
            "schema validation",
            "training",
            "model-quality validation",
            "Evidently",
            "MLflow registration",
            "alias verification",
            "model artifact saving",
            "service reload",
            "serving verification",
            "total pipeline",
        ],
    )


def test_main_allows_validation_failed_result(monkeypatch):
    from training.run_pipeline import main

    monkeypatch.setattr(
        "training.run_pipeline.run_training_pipeline",
        lambda: {"status": "validation_failed", "errors": ["bad data"]},
    )

    main()


def test_main_logs_sanitized_pipeline_result(monkeypatch, caplog):
    from training.run_pipeline import main

    secret = "super-secret"
    result = {
        "status": "success",
        "reload": {
            "status": "model_reloaded",
            "message": f"accepted with {secret}",
        },
    }

    monkeypatch.setenv("SERVICE_RELOAD_SECRET", secret)
    monkeypatch.setattr("training.run_pipeline.run_training_pipeline", lambda: result)

    with caplog.at_level(logging.INFO):
        main()

    completed_records = [
        record
        for record in caplog.records
        if record.msg == "Training pipeline completed: %s"
    ]

    assert result["reload"]["message"] == f"accepted with {secret}"
    assert len(completed_records) == 1
    assert completed_records[0].args == {
        "status": "success",
        "reload": {
            "status": "model_reloaded",
            "message": "accepted with [REDACTED]",
        },
    }
    assert "Training pipeline completed:" in caplog.text
    assert secret not in caplog.text
    assert "[REDACTED]" in caplog.text


def test_main_does_not_train_or_register_on_bad_downloaded_data(tmp_path, monkeypatch):
    from training.run_pipeline import main

    bad_csv_path = tmp_path / "bad.csv"
    validation_report_path = tmp_path / "validation_report.json"

    monkeypatch.setenv("DATA_URL", "https://example.com/bad.csv")
    monkeypatch.setattr("training.run_pipeline._DEFAULT_DATA_PATH", str(bad_csv_path))
    monkeypatch.setattr(
        "training.run_pipeline._DEFAULT_VALIDATION_REPORT_PATH",
        str(validation_report_path),
    )

    def fake_download_data(data_url, output_path):
        assert data_url == "https://example.com/bad.csv"
        Path(output_path).write_text("Password\npassword123\n", encoding="utf-8")
        return str(output_path)

    def fake_validate_data_file(input_path, report_path):
        assert input_path == str(bad_csv_path)
        assert report_path == str(validation_report_path)
        assert "Times" not in Path(input_path).read_text(encoding="utf-8")
        report = {
            "is_valid": False,
            "errors": ["DataFrame is missing required columns: Times"],
            "n_rows": 1,
            "columns": ["Password"],
            "schema_metrics": {
                "n_rows": 1,
                "columns": ["Password"],
                "thresholds": SCHEMA_THRESHOLDS,
            },
            "thresholds": SCHEMA_THRESHOLDS,
        }
        Path(report_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return SimpleNamespace(
            is_valid=False,
            errors=report["errors"],
            n_rows=1,
            columns=["Password"],
            metrics=report["schema_metrics"],
            thresholds=SCHEMA_THRESHOLDS,
        )

    def fail_if_called(*args, **kwargs):
        pytest.fail("training or registration side effect must not be called")

    monkeypatch.setattr("training.run_pipeline.download_data", fake_download_data)
    monkeypatch.setattr(
        "training.run_pipeline.validate_data_file",
        fake_validate_data_file,
    )
    monkeypatch.setattr("training.run_pipeline.train_password_model", fail_if_called)
    monkeypatch.setattr(
        "training.run_pipeline.register_model_in_mlflow",
        fail_if_called,
    )
    monkeypatch.setattr(
        "training.run_pipeline.call_reload_model_endpoint",
        fail_if_called,
    )

    main()

    assert validation_report_path.exists()
    validation_report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    assert validation_report["is_valid"] is False
    assert validation_report["errors"]
    assert validation_report["n_rows"] == 1
    assert validation_report["columns"] == ["Password"]
    assert validation_report["schema_metrics"]["thresholds"] == SCHEMA_THRESHOLDS
    assert validation_report["thresholds"] == SCHEMA_THRESHOLDS


def test_run_training_pipeline_uses_scored_df_for_evidently_after_validation(
    monkeypatch,
):
    from training.run_pipeline import run_training_pipeline

    training_df = FakeTrainingFrame()
    scored_df = object()
    calls = []

    monkeypatch.setenv("DATA_URL", "https://example.com/data.csv")
    monkeypatch.setattr(
        "training.run_pipeline.download_data",
        lambda data_url, output_path: "downloaded.csv",
    )
    monkeypatch.setattr("training.run_pipeline.read_csv", lambda path: object())
    monkeypatch.setattr(
        "training.run_pipeline.validate_data_file",
        lambda input_path, report_path: ValidationResult(
            True, [], 1, ["Password", "Times"]
        ),
    )
    monkeypatch.setattr(
        "training.run_pipeline.validate_password_dataframe",
        lambda df: (True, [], training_df, None),
    )

    def validate_quality(df):
        calls.append(("model_quality", df))
        return SimpleNamespace(
            is_valid=True,
            errors=[],
            metrics={"rmse": 0.1},
            scored_df=scored_df,
        )

    def run_evidently(df, output_path):
        calls.append(("evidently", df, output_path))
        return {"status": "failure"}

    monkeypatch.setattr(
        "training.run_pipeline.validate_model_quality_with_prod_model",
        validate_quality,
    )
    monkeypatch.setattr("training.run_pipeline.run_evidently_tests", run_evidently)
    monkeypatch.setattr(
        "training.run_pipeline.train_password_model",
        lambda df: calls.append(("train", df)) or ("model", {"n_rows": 1}),
    )
    monkeypatch.setattr(
        "training.run_pipeline.save_model_artifact",
        lambda model, output_path: "artifacts/model.joblib",
    )
    monkeypatch.setattr(
        "training.run_pipeline.register_model_in_mlflow",
        lambda model, metrics, validation_report=None, model_alias=None: {
            "model_name": "passwords",
            "model_alias": model_alias,
            "model_version": "1",
            "alias_verified": True,
            "verified_model_version": "1",
        },
    )
    monkeypatch.setattr(
        "training.run_pipeline.call_reload_model_endpoint",
        lambda registration_result=None: {"status": "skipped"},
    )

    result = run_training_pipeline()

    assert result["evidently_report"] == {"status": "failure"}
    assert calls[:3] == [
        ("train", training_df),
        ("model_quality", training_df),
        ("evidently", scored_df, "reports/tests.json"),
    ]


def test_run_training_pipeline_continues_when_evidently_raises(
    tmp_path, monkeypatch, caplog
):
    from training.run_pipeline import run_training_pipeline

    reports_path = tmp_path / "tests.json"
    training_df = FakeTrainingFrame()
    calls = []

    monkeypatch.setenv("DATA_URL", "https://example.com/data.csv")
    monkeypatch.setattr(
        "training.run_pipeline._DEFAULT_EVIDENTLY_REPORT_PATH", str(reports_path)
    )
    monkeypatch.setattr(
        "training.run_pipeline.download_data",
        lambda data_url, output_path: "downloaded.csv",
    )
    monkeypatch.setattr("training.run_pipeline.read_csv", lambda path: object())
    monkeypatch.setattr(
        "training.run_pipeline.validate_data_file",
        lambda input_path, report_path: ValidationResult(
            True, [], 1, ["Password", "Times"]
        ),
    )
    monkeypatch.setattr(
        "training.run_pipeline.validate_password_dataframe",
        lambda df: (True, [], training_df, None),
    )
    monkeypatch.setattr(
        "training.run_pipeline.validate_model_quality_with_prod_model",
        lambda df: SimpleNamespace(
            is_valid=True,
            errors=[],
            metrics={"rmse": 0.1},
            scored_df=None,
        ),
    )

    def fail_evidently(df, output_path):
        raise RuntimeError("evidently backend unavailable")

    monkeypatch.setattr("training.run_pipeline.run_evidently_tests", fail_evidently)
    monkeypatch.setattr(
        "training.run_pipeline.train_password_model",
        lambda df: calls.append("train") or ("model", {"n_rows": 1}),
    )
    monkeypatch.setattr(
        "training.run_pipeline.save_model_artifact",
        lambda model, output_path: "artifacts/model.joblib",
    )
    monkeypatch.setattr(
        "training.run_pipeline.register_model_in_mlflow",
        lambda model, metrics, validation_report=None, model_alias=None: {
            "model_name": "passwords",
            "model_alias": model_alias,
            "model_version": "1",
            "alias_verified": True,
            "verified_model_version": "1",
        },
    )
    monkeypatch.setattr(
        "training.run_pipeline.call_reload_model_endpoint",
        lambda registration_result=None: {"status": "skipped"},
    )

    with caplog.at_level(logging.WARNING):
        result = run_training_pipeline()

    assert calls == ["train"]
    assert result["evidently_report"]["status"] == "warning"
    assert result["evidently_report"]["evidently_failed"] is True
    assert (
        json.loads(reports_path.read_text(encoding="utf-8"))
        == result["evidently_report"]
    )
    assert "continuing pipeline" in caplog.text


def test_run_training_pipeline_does_not_reload_when_registration_fails(monkeypatch):
    from training.run_pipeline import run_training_pipeline

    calls = []

    training_df = FakeTrainingFrame()

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
        lambda df: (True, [], training_df, None),
    )
    monkeypatch.setattr(
        "training.run_pipeline.validate_model_quality_with_prod_model",
        lambda df: SimpleNamespace(
            is_valid=True,
            errors=[],
            metrics={"rmse": 0.1},
            scored_df=None,
        ),
    )
    monkeypatch.setattr(
        "training.run_pipeline.train_password_model",
        lambda df: ("model", {"n_rows": 1}),
    )
    monkeypatch.setattr(
        "training.run_pipeline.save_model_artifact",
        lambda model, output_path: "artifacts/model.joblib",
    )

    def fail_registration(model, metrics, validation_report=None, model_alias=None):
        calls.append("register")
        raise RuntimeError("mlflow is down")

    monkeypatch.setattr(
        "training.run_pipeline.register_model_in_mlflow",
        fail_registration,
    )
    monkeypatch.setattr(
        "training.run_pipeline.call_reload_model_endpoint",
        lambda *_args, **_kwargs: calls.append("reload"),
    )

    with pytest.raises(RuntimeError, match="mlflow is down"):
        run_training_pipeline()

    assert calls == ["register"]


def test_run_training_pipeline_reloads_after_prod_registration(monkeypatch, caplog):
    from training.run_pipeline import run_training_pipeline

    calls = []
    schema_metrics = {
        "row_count": 2,
        "null_passwords": 0,
        "thresholds": SCHEMA_THRESHOLDS,
    }
    model_quality_metrics = {"rmse": 0.1, "mae": 0.05}
    training_df = FakeDataFrame(
        {
            "Password": ["alpha", "beta"],
            "Times": [10, 20],
            "source_row": [1, 2],
        }
    )
    scored_df = training_df.assign(
        target_log=[1.0, 2.0], prediction=[1.1, 2.1], prediction_error=[0.1, 0.1]
    )
    merged_validation_report = {
        "is_valid": True,
        "errors": [],
        "n_rows": 2,
        "columns": ["Password", "Times"],
        "schema_metrics": schema_metrics,
        "model_quality_is_valid": True,
        "model_quality_errors": [],
        "model_quality_metrics": model_quality_metrics,
        "thresholds": SCHEMA_THRESHOLDS,
    }
    registration_result = {
        "model_name": "passwords",
        "model_alias": "prod",
        "model_version": "1",
        "model_uri": "models:/passwords/1",
        "alias_verified": True,
        "verified_model_version": "1",
    }

    monkeypatch.setenv("DATA_URL", "https://example.com/data.csv")
    monkeypatch.setattr(
        "training.run_pipeline.download_data",
        lambda data_url, output_path: "downloaded.csv",
    )
    monkeypatch.setattr(
        "training.run_pipeline.validate_data_file",
        lambda input_path, report_path: SimpleNamespace(
            is_valid=True,
            errors=[],
            n_rows=2,
            columns=["Password", "Times"],
            metrics=schema_metrics,
            cleaned_df=training_df,
            thresholds=SCHEMA_THRESHOLDS,
        ),
    )

    def validate_quality(df):
        calls.append(("model_quality", df))
        return ModelQualityValidationResult(
            True, [], model_quality_metrics, scored_df=scored_df
        )

    def run_evidently(df, output_path):
        calls.append(("evidently", df, output_path))
        return {"status": "success"}

    def train(df):
        calls.append(("train", df.copy()))
        return "model", {"n_rows": len(df.index)}

    def save(model, output_path):
        calls.append(("save", model, output_path))
        return "artifacts/model.joblib"

    def register(model, metrics, validation_report=None, model_alias=None):
        calls.append(("register", model, metrics, validation_report, model_alias))
        return registration_result

    def verify_alias(result):
        calls.append(("verify_alias", result))

    def reload_model(result=None):
        calls.append(("reload", result))
        return {"status": "model_reloaded", "token": "super-secret"}

    def verify_serving(registration_result, reload_result, serving_training_df):
        calls.append(
            ("verify_serving", registration_result, reload_result, serving_training_df)
        )
        return {
            "status": "verified",
            "loaded_model_version": registration_result["model_version"],
        }

    monkeypatch.setattr(
        "training.run_pipeline.validate_model_quality_with_prod_model",
        validate_quality,
    )
    monkeypatch.setattr("training.run_pipeline.run_evidently_tests", run_evidently)
    monkeypatch.setattr("training.run_pipeline.train_password_model", train)
    monkeypatch.setattr("training.run_pipeline.save_model_artifact", save)
    monkeypatch.setattr("training.run_pipeline.register_model_in_mlflow", register)
    monkeypatch.setattr(
        "training.run_pipeline._ensure_registration_alias_verified", verify_alias
    )
    monkeypatch.setattr(
        "training.run_pipeline.call_reload_model_endpoint", reload_model
    )
    monkeypatch.setattr(
        "training.run_pipeline.verify_serving_after_reload", verify_serving
    )

    monkeypatch.setenv("SERVICE_RELOAD_SECRET", "super-secret")

    with caplog.at_level(logging.INFO):
        result = run_training_pipeline()

    assert [call[0] for call in calls] == [
        "train",
        "model_quality",
        "evidently",
        "register",
        "verify_alias",
        "save",
        "reload",
        "verify_serving",
    ]
    assert list(calls[0][1].columns) == ["Password", "Times"]
    assert calls[0][1].equals(training_df[["Password", "Times"]])
    assert calls[1][0] == "model_quality"
    assert calls[1][1] is training_df
    assert calls[2][0] == "evidently"
    assert calls[2][1] is scored_df
    assert calls[2][2] == "reports/tests.json"
    assert calls[3] == (
        "register",
        "model",
        {"n_rows": 2},
        merged_validation_report,
        "prod",
    )
    assert calls[4] == ("verify_alias", registration_result)
    assert calls[5] == ("save", "model", "artifacts/model.joblib")
    assert calls[6] == ("reload", registration_result)
    assert calls[7] == (
        "verify_serving",
        registration_result,
        {"status": "model_reloaded", "token": "super-secret"},
        training_df,
    )
    assert result["validation_report"] == merged_validation_report
    assert result["reload"] == {"status": "model_reloaded", "token": "super-secret"}
    assert result["serving_verification"] == {
        "status": "verified",
        "loaded_model_version": "1",
    }
    assert "data downloaded: downloaded.csv" in caplog.text
    assert "validation passed" in caplog.text
    assert "model trained" in caplog.text
    assert (
        "model registered: {'model_name': 'passwords', 'model_alias': 'prod', 'model_version': '1', 'model_uri': 'models:/passwords/1', 'alias_verified': True, 'verified_model_version': '1'}"
        in caplog.text
    )
    assert (
        "service reload response received: {'status': 'model_reloaded', 'token': '[REDACTED]'}"
        in caplog.text
    )
    assert "super-secret" not in caplog.text
    _assert_duration_logs(
        caplog,
        [
            "data download",
            "schema validation",
            "training",
            "model-quality validation",
            "Evidently",
            "MLflow registration",
            "alias verification",
            "model artifact saving",
            "service reload",
            "serving verification",
            "total pipeline",
        ],
    )


def test_two_valid_datasets_create_versions_and_move_prod_alias(monkeypatch):
    from training.run_pipeline import run_training_pipeline

    datasets = {
        "downloaded-a.csv": FakeDataFrame(
            {"Password": ["alpha", "bravo"], "Times": [10, 20]}
        ),
        "downloaded-b.csv": FakeDataFrame(
            {"Password": ["charlie", "delta"], "Times": [30, 40]}
        ),
    }
    data_paths_by_url = {
        "https://example.com/a.csv": "downloaded-a.csv",
        "https://example.com/b.csv": "downloaded-b.csv",
    }
    registered_versions = []
    prod_alias_versions = []

    monkeypatch.setattr(
        "training.run_pipeline.download_data",
        lambda data_url, output_path: data_paths_by_url[data_url],
    )

    def validate_data(input_path, report_path):
        return SimpleNamespace(
            is_valid=True,
            errors=[],
            n_rows=len(datasets[input_path].index),
            columns=["Password", "Times"],
            metrics={"row_count": len(datasets[input_path].index)},
            cleaned_df=datasets[input_path],
        )

    def validate_quality(df):
        return ModelQualityValidationResult(
            False,
            ["old prod rmse is above threshold"],
            {"rmse": 9.0, "mae": 8.0},
            scored_df=df,
        )

    def train(df):
        return f"model-{df['Password'][0]}", {"n_rows": len(df.index)}

    def register(model, metrics, validation_report=None, model_alias=None):
        version = str(len(registered_versions) + 1)
        registered_versions.append(
            {
                "version": version,
                "model": model,
                "metrics": metrics,
                "validation_report": validation_report,
                "alias": model_alias,
            }
        )
        prod_alias_versions.append(version)
        return {
            "model_name": "passwords",
            "model_alias": model_alias,
            "model_version": version,
            "model_uri": f"models:/passwords/{version}",
            "alias_verified": True,
            "verified_model_version": version,
        }

    monkeypatch.setattr("training.run_pipeline.validate_data_file", validate_data)
    monkeypatch.setattr(
        "training.run_pipeline.validate_model_quality_with_prod_model",
        validate_quality,
    )
    monkeypatch.setattr(
        "training.run_pipeline.run_evidently_tests",
        lambda df, output_path: {"status": "success"},
    )
    monkeypatch.setattr("training.run_pipeline.train_password_model", train)
    monkeypatch.setattr(
        "training.run_pipeline.save_model_artifact",
        lambda model, output_path: "artifacts/model.joblib",
    )
    monkeypatch.setattr("training.run_pipeline.register_model_in_mlflow", register)
    monkeypatch.setattr(
        "training.run_pipeline.call_reload_model_endpoint",
        lambda registration_result=None: {"status": "skipped"},
    )

    first_result = run_training_pipeline("https://example.com/a.csv")
    second_result = run_training_pipeline("https://example.com/b.csv")

    assert [
        result["registration"]["model_version"]
        for result in [first_result, second_result]
    ] == ["1", "2"]
    assert [registration["alias"] for registration in registered_versions] == [
        "prod",
        "prod",
    ]
    assert prod_alias_versions == ["1", "2"]
    assert [registration["model"] for registration in registered_versions] == [
        "model-alpha",
        "model-charlie",
    ]
    assert all(
        registration["validation_report"]["is_valid"] is True
        and registration["validation_report"]["model_quality_is_valid"] is False
        for registration in registered_versions
    )


def test_run_training_pipeline_requires_reload_url_in_ci_after_registration(
    monkeypatch,
):
    from training.run_pipeline import run_training_pipeline

    training_df = FakeTrainingFrame()

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
        lambda df: (True, [], training_df, None),
    )
    monkeypatch.setattr(
        "training.run_pipeline.validate_model_quality_with_prod_model",
        lambda df: SimpleNamespace(
            is_valid=True,
            errors=[],
            metrics={"rmse": 0.1},
            scored_df=None,
        ),
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
        lambda model, metrics, validation_report=None, model_alias=None: {
            "model_name": "passwords",
            "model_alias": model_alias,
            "model_version": "1",
            "model_uri": "models:/passwords/1",
            "alias_verified": True,
            "verified_model_version": "1",
        },
    )

    with pytest.raises(RuntimeError, match="SERVICE_RELOAD_URL is required in CI"):
        run_training_pipeline()


def test_registration_alias_gate_rejects_unverified_alias_before_reload():
    from training.run_pipeline import _ensure_registration_alias_verified

    with pytest.raises(RuntimeError, match="refusing to reload service"):
        _ensure_registration_alias_verified(
            {
                "model_name": "passwords",
                "model_alias": "prod",
                "model_version": "2",
                "alias_verified": False,
                "verified_model_version": "1",
            }
        )


def test_sanitize_for_log_redacts_all_configured_secret_env_values(monkeypatch):
    from training.run_pipeline import _sanitize_for_log

    secrets = {
        "AWS_ACCESS_KEY_ID": "aws-access-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret-key",
        "MLFLOW_TRACKING_PASSWORD": "mlflow-password",
        "DOCKERHUB_TOKEN": "dockerhub-token",
        "SERVICE_RELOAD_SECRET": "reload-secret",
        "SERVICE_RELOAD_URL": "https://reload-user:reload-password@service.example/reload_model",
        "GITLAB_TRIGGER_TOKEN": "gitlab-trigger-token",
        "AMVERA_PASSWORD": "amvera-password",
    }
    for env_name, secret in secrets.items():
        monkeypatch.setenv(env_name, secret)

    value = {
        "message": " ".join(secrets.values()),
        "items": [
            f"Authorization: Bearer {secrets['GITLAB_TRIGGER_TOKEN']}",
            (f"password={secrets['AMVERA_PASSWORD']}",),
        ],
        "callback_url": "https://user:inline-password@example.com/reload_model",
        "headers": {"X-Service-Token": secrets["SERVICE_RELOAD_SECRET"]},
    }

    sanitized = _sanitize_for_log(value)
    sanitized_text = repr(sanitized)

    for secret in secrets.values():
        assert secret not in sanitized_text
    assert "inline-password" not in sanitized_text
    assert sanitized["headers"]["X-Service-Token"] == "[REDACTED]"
    assert "[REDACTED]" in sanitized_text


def test_main_completed_log_redacts_all_configured_secret_env_values(
    monkeypatch, caplog
):
    from training.run_pipeline import main

    secrets = {
        "AWS_ACCESS_KEY_ID": "aws-access-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret-key",
        "MLFLOW_TRACKING_PASSWORD": "mlflow-password",
        "DOCKERHUB_TOKEN": "dockerhub-token",
        "SERVICE_RELOAD_SECRET": "reload-secret",
        "SERVICE_RELOAD_URL": "https://reload-user:reload-password@service.example/reload_model",
        "GITLAB_TRIGGER_TOKEN": "gitlab-trigger-token",
        "AMVERA_PASSWORD": "amvera-password",
    }
    for env_name, secret in secrets.items():
        monkeypatch.setenv(env_name, secret)

    result = {
        "status": "success",
        "registration": {
            "mlflow_url": f"https://mlflow-user:{secrets['MLFLOW_TRACKING_PASSWORD']}@mlflow.example",
            "access_key": secrets["AWS_ACCESS_KEY_ID"],
        },
        "reload": {
            "url": secrets["SERVICE_RELOAD_URL"],
            "message": f"accepted {secrets['SERVICE_RELOAD_SECRET']}",
        },
        "serving_verification": {
            "health": {"last_reload_error": None},
            "predict": {
                "url": "https://predict-user:predict-password@service.example/predict"
            },
        },
        "tokens": list(secrets.values()),
    }
    monkeypatch.setattr("training.run_pipeline.run_training_pipeline", lambda: result)

    with caplog.at_level(logging.INFO):
        main()

    assert "Training pipeline completed:" in caplog.text
    for secret in secrets.values():
        assert secret not in caplog.text
    assert "predict-password" not in caplog.text
    assert "[REDACTED]" in caplog.text
