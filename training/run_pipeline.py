import json
import logging
import math
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse
import pandas as pd
import requests

_RELOAD_ATTEMPTS = 3
_RELOAD_RETRY_DELAY_SECONDS = 10
_RELOAD_TIMEOUT_SECONDS = 120
_SERVING_CHECK_TIMEOUT_SECONDS = 30
_SERVING_VERIFICATION_SAMPLE_SIZE = 20
_SERVING_VERIFICATION_MIN_SAMPLE_SIZE = 20
_SERVING_VERIFICATION_MAX_SAMPLE_SIZE = 50
_SERVING_VERIFICATION_ROUNDS = 3
_SERVING_PREDICT_MAX_ABS_ERROR_TOLERANCE = 0.5
_DEFAULT_DATA_PATH = "artifacts/new_data.csv"
_DEFAULT_MODEL_ARTIFACT_PATH = "artifacts/model.joblib"
_DEFAULT_VALIDATION_REPORT_PATH = "validation_reports/validation_report.json"
_DEFAULT_EVIDENTLY_REPORT_PATH = "reports/tests.json"
_PRODUCTION_MODEL_ALIAS = "prod"
_FAIL_CI_ON_SERVING_VERIFICATION_FAILED_ENV = "FAIL_CI_ON_SERVING_VERIFICATION_FAILED"


logger = logging.getLogger(__name__)

_SENSITIVE_LOG_KEY_PARTS = ("secret", "credential", "password", "token", "key")
_SENSITIVE_ENV_NAMES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "MLFLOW_TRACKING_PASSWORD",
    "DOCKERHUB_TOKEN",
    "SERVICE_RELOAD_SECRET",
    "SERVICE_RELOAD_URL",
    "GITLAB_TRIGGER_TOKEN",
    "AMVERA_PASSWORD",
)
_REDACTED_LOG_VALUE = "[REDACTED]"
_URL_CREDENTIALS_PATTERN = re.compile(r"(://)([^/\s:@]+):([^@/\s]+)@")


def _is_sensitive_log_key(key: str) -> bool:
    normalized_key = key.lower()
    return any(part in normalized_key for part in _SENSITIVE_LOG_KEY_PARTS)


def _secret_env_values() -> tuple[str, ...]:
    """Collect configured secret values that must never appear in logs."""
    return tuple(
        value for env_name in _SENSITIVE_ENV_NAMES if (value := os.getenv(env_name))
    )


def _sanitize_string_for_log(value: str) -> str:
    sanitized = value
    for secret_value in _secret_env_values():
        sanitized = sanitized.replace(secret_value, _REDACTED_LOG_VALUE)

    return _URL_CREDENTIALS_PATTERN.sub(
        rf"\1{_REDACTED_LOG_VALUE}:{_REDACTED_LOG_VALUE}@", sanitized
    )


def _sanitize_for_log(value: Any) -> Any:
    """Return a log-safe copy of a value by redacting known sensitive data."""
    if isinstance(value, dict):
        return {
            key: (
                _REDACTED_LOG_VALUE
                if _is_sensitive_log_key(str(key))
                else _sanitize_for_log(item)
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_sanitize_for_log(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_sanitize_for_log(item) for item in value)

    if isinstance(value, str):
        return _sanitize_string_for_log(value)

    return value


def _timed_step(step_name: str, step_callable, *args, **kwargs):
    """Run a pipeline step and log its elapsed wall-clock duration."""
    started_at = time.monotonic()
    try:
        return step_callable(*args, **kwargs)
    finally:
        logger.info("%s took %.3f sec", step_name, time.monotonic() - started_at)


def download_data(data_url: str, output_path: str) -> str:
    from training.download_data import download_data as download_data_impl

    return download_data_impl(data_url, output_path)


def read_csv(path: str):
    import pandas as pd

    return pd.read_csv(path)


def run_evidently_tests(df, output_path: str = _DEFAULT_EVIDENTLY_REPORT_PATH) -> dict:
    from training.validate_data import run_evidently_tests as run_evidently_tests_impl

    return run_evidently_tests_impl(df, output_path)


def validate_data_file(
    input_path: str, report_path: str = _DEFAULT_VALIDATION_REPORT_PATH
):
    from training.validate_data import validate_data_file as validate_data_file_impl

    return validate_data_file_impl(input_path, report_path)


def validate_password_dataframe(df):
    from training.validate_data import (
        validate_password_dataframe as validate_password_dataframe_impl,
    )

    return validate_password_dataframe_impl(df)


def validate_model_quality_with_prod_model(df):
    from training.model_quality_validation import (
        validate_model_quality_with_prod_model as impl,
    )

    return impl(df)


def train_password_model(df):
    from training.train_model import train_password_model as train_password_model_impl

    return train_password_model_impl(df)


def save_model_artifact(model, output_path: str = _DEFAULT_MODEL_ARTIFACT_PATH) -> str:
    from training.train_model import save_model_artifact as save_model_artifact_impl

    return save_model_artifact_impl(model, output_path)


def register_model_in_mlflow(
    model,
    metrics: dict,
    validation_report: dict | None = None,
    model_alias: str | None = None,
) -> dict:
    from training.register_model import register_model_in_mlflow as register_model_impl

    return register_model_impl(
        model,
        metrics,
        validation_report=validation_report,
        model_alias=model_alias,
    )


def _ensure_registration_alias_verified(registration_result: dict[str, Any]) -> None:
    """Require MLflow alias read-back confirmation before service reload."""
    model_version = str(registration_result.get("model_version"))
    verified_model_version = str(registration_result.get("verified_model_version"))

    if (
        registration_result.get("alias_verified") is not True
        or verified_model_version != model_version
    ):
        raise RuntimeError(
            "MLflow registration did not confirm that alias "
            f"{registration_result.get('model_alias')!r} points to model version "
            f"{model_version!r}; refusing to reload service."
        )


def _service_endpoint_url(explicit_env_name: str, fallback_path: str) -> str | None:
    """Resolve a serving endpoint URL from env or from SERVICE_RELOAD_URL."""
    explicit_url = os.getenv(explicit_env_name)
    if explicit_url:
        return explicit_url

    reload_url = os.getenv("SERVICE_RELOAD_URL")
    if not reload_url:
        return None

    parsed_url = urlparse(reload_url)
    if not parsed_url.scheme or not parsed_url.netloc:
        return urljoin(reload_url.rstrip("/") + "/", fallback_path.lstrip("/"))

    return urlunparse(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            fallback_path,
            "",
            "",
            "",
        )
    )


def _split_csv_env(value: str | None) -> list[str]:
    if not value:
        return []

    return [item.strip() for item in value.split(",") if item.strip()]


def _service_state_urls() -> list[str]:
    """Return state endpoints that must be checked after reload.

    SERVICE_REPLICA_STATE_URLS may contain comma-separated direct /model_state or
    /health URLs for every serving replica. When it is not set, the current
    Amvera deployment is treated as a single instance and SERVICE_HEALTH_URL (or
    SERVICE_RELOAD_URL-derived /model_state) is checked.
    """
    replica_urls = _split_csv_env(os.getenv("SERVICE_REPLICA_STATE_URLS"))
    if replica_urls:
        return replica_urls

    health_url = _service_endpoint_url("SERVICE_HEALTH_URL", "/model_state")
    return [health_url] if health_url else []


def _expected_model_version(registration_result: dict[str, Any] | None) -> str | None:
    if registration_result is None:
        return None

    model_version = registration_result.get("model_version")
    if model_version is None:
        return None

    return str(model_version)


def _env_flag_is_true(env_name: str, default: bool = False) -> bool:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int_env(env_name: str, default_value: int) -> int:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default_value

    try:
        return int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"{env_name} must be an integer, got {raw_value!r}."
        ) from exc


def _parse_float_env(env_name: str, default_value: float) -> float:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default_value

    try:
        return float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{env_name} must be a float, got {raw_value!r}.") from exc


def _serving_verification_sample(training_df: pd.DataFrame) -> pd.DataFrame:
    sample_size = _parse_int_env(
        "SERVICE_PREDICT_CHECK_ROWS", _SERVING_VERIFICATION_SAMPLE_SIZE
    )
    sample_size = max(
        _SERVING_VERIFICATION_MIN_SAMPLE_SIZE,
        min(_SERVING_VERIFICATION_MAX_SAMPLE_SIZE, sample_size),
    )

    if not {"Password", "Times"}.issubset(training_df.columns):
        raise RuntimeError(
            "training_df must contain Password and Times columns for serving "
            "prediction verification."
        )

    check_df = training_df[["Password", "Times"]].dropna().copy()
    check_df["Times"] = pd.to_numeric(check_df["Times"], errors="coerce")
    check_df = check_df.dropna(subset=["Password", "Times"])
    check_df = check_df[check_df["Times"] >= 0]
    if check_df.empty:
        raise RuntimeError(
            "training_df does not contain any valid Password/Times rows for "
            "serving prediction verification."
        )

    row_count = min(sample_size, len(check_df.index))
    return check_df.head(row_count).reset_index(drop=True)


def _assert_serving_state(
    health_state: dict[str, Any], state_url: str, expected_version: str | None
) -> None:
    instance_id = health_state.get("instance_id") or state_url
    if health_state.get("model_loaded") is not True:
        raise RuntimeError(
            "Serving state check for instance "
            f"{instance_id!r} reports model_loaded="
            f"{health_state.get('model_loaded')!r}."
        )
    if health_state.get("last_reload_status") != "success":
        raise RuntimeError(
            "Serving state check for instance "
            f"{instance_id!r} reports last_reload_status="
            f"{health_state.get('last_reload_status')!r}."
        )
    if (
        expected_version is not None
        and health_state.get("loaded_version") != expected_version
    ):
        raise RuntimeError(
            "Serving state check for instance "
            f"{instance_id!r} loaded unexpected model version: "
            f"{health_state.get('loaded_version')!r} != {expected_version!r}."
        )


def _validate_predict_response(
    predict_result: dict[str, Any], passwords: list[str], expected_targets: list[float]
) -> tuple[list[float], float]:
    predictions = predict_result.get("Times")
    if not isinstance(predictions, list) or len(predictions) != len(passwords):
        raise RuntimeError(
            "Serving predict check returned invalid Times: "
            f"{_sanitize_for_log(predictions)!r}."
        )

    prediction_values: list[float] = []
    for prediction in predictions:
        try:
            prediction_value = float(prediction)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Serving predict check returned non-numeric Times: "
                f"{_sanitize_for_log(predictions)!r}."
            ) from exc
        if not math.isfinite(prediction_value):
            raise RuntimeError(
                "Serving predict check returned non-finite Times: "
                f"{_sanitize_for_log(predictions)!r}."
            )
        prediction_values.append(prediction_value)

    max_abs_error = max(
        abs(prediction - expected)
        for prediction, expected in zip(prediction_values, expected_targets)
    )
    return prediction_values, float(max_abs_error)


def _request_json_with_retries(method: str, url: str, **kwargs) -> dict[str, Any]:
    last_error: str | None = None

    for attempt in range(1, _RELOAD_ATTEMPTS + 1):
        try:
            response = requests.request(
                method,
                url,
                timeout=_SERVING_CHECK_TIMEOUT_SECONDS,
                **kwargs,
            )
            response.raise_for_status()
            try:
                body = response.json()
            except ValueError as exc:
                raise RuntimeError(f"{url} returned a non-JSON response") from exc

            if not isinstance(body, dict):
                raise RuntimeError(f"{url} returned a non-object JSON response")

            return body
        except (requests.RequestException, RuntimeError) as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            sanitized_error = _sanitize_for_log(str(exc))
            if status_code is None:
                last_error = type(exc).__name__
                if sanitized_error:
                    last_error = f"{last_error}: {sanitized_error}"
            else:
                last_error = f"{type(exc).__name__} with HTTP status {status_code}"

            if attempt < _RELOAD_ATTEMPTS:
                time.sleep(_RELOAD_RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"Serving check {method} {_sanitize_for_log(url)} failed after "
        f"{_RELOAD_ATTEMPTS} attempts. Last error: {_sanitize_for_log(last_error)}."
    )


def verify_serving_after_reload(
    registration_result: dict[str, Any] | None,
    reload_result: dict[str, Any],
    training_df: pd.DataFrame,
) -> dict[str, Any]:
    """Verify that the serving app exposes the reloaded model and predicts."""
    if reload_result.get("status") == "skipped":
        return {
            "status": "skipped",
            "reason": "service reload was skipped",
        }

    expected_version = _expected_model_version(registration_result)
    state_urls = _service_state_urls()
    predict_url = _service_endpoint_url("SERVICE_PREDICT_URL", "/predict")
    if not state_urls or not predict_url:
        raise RuntimeError(
            "SERVICE_REPLICA_STATE_URLS or SERVICE_HEALTH_URL, and "
            "SERVICE_PREDICT_URL are required for serving verification when "
            "SERVICE_RELOAD_URL cannot be used to derive them."
        )

    sample_df = _serving_verification_sample(training_df)
    sample_passwords = [str(password) for password in sample_df["Password"].tolist()]
    expected_targets = [
        float(math.log10(float(times) + 1.0)) for times in sample_df["Times"].tolist()
    ]
    tolerance = _parse_float_env(
        "SERVICE_PREDICT_MAX_ABS_ERROR_TOLERANCE",
        _SERVING_PREDICT_MAX_ABS_ERROR_TOLERANCE,
    )
    if tolerance < 0:
        raise RuntimeError("SERVICE_PREDICT_MAX_ABS_ERROR_TOLERANCE must be >= 0.")

    rounds = max(
        1,
        _parse_int_env(
            "SERVICE_SERVING_VERIFICATION_ROUNDS", _SERVING_VERIFICATION_ROUNDS
        ),
    )

    replica_checks: list[dict[str, Any]] = []
    latest_replica_states_by_url: dict[str, dict[str, Any]] = {}
    predict_checks: list[dict[str, Any]] = []
    max_abs_error = 0.0

    for round_number in range(1, rounds + 1):
        for state_url in state_urls:
            health_state = _request_json_with_retries("GET", state_url)
            _assert_serving_state(health_state, state_url, expected_version)

            replica_state = {
                "round": round_number,
                "url": state_url,
                "state": health_state,
            }
            replica_checks.append(replica_state)
            latest_replica_states_by_url[state_url] = replica_state

        predict_result = _request_json_with_retries(
            "POST",
            predict_url,
            json={"Password": sample_passwords},
        )
        _, round_max_abs_error = _validate_predict_response(
            predict_result, sample_passwords, expected_targets
        )
        max_abs_error = max(max_abs_error, round_max_abs_error)
        predict_checks.append(
            {
                "round": round_number,
                "url": predict_url,
                "request_count": len(sample_passwords),
                "response_count": len(sample_passwords),
                "max_abs_error": round_max_abs_error,
            }
        )

        logger.info(
            "serving predict check round %s/%s: checked_rows=%s "
            "max_abs_error=%.6f tolerance=%.6f sample_passwords=%s",
            round_number,
            rounds,
            len(sample_passwords),
            round_max_abs_error,
            tolerance,
            _sanitize_for_log(sample_passwords[:5]),
        )

        if round_max_abs_error > tolerance:
            raise RuntimeError(
                "Serving predict check exceeded max_abs_error tolerance: "
                f"{round_max_abs_error:.6f} > {tolerance:.6f}; "
                f"checked_rows={len(sample_passwords)}; "
                f"sample_passwords={_sanitize_for_log(sample_passwords[:5])!r}."
            )

    ordered_unique_state_urls = list(dict.fromkeys(state_urls))
    replica_states = [
        latest_replica_states_by_url[state_url]
        for state_url in ordered_unique_state_urls
    ]
    health_state = replica_states[-1]["state"]

    return {
        "status": "verified",
        "expected_model_version": expected_version,
        "loaded_model_version": health_state.get("loaded_version"),
        "replica_count": len(replica_states),
        "replicas": replica_states,
        "replica_checks": replica_checks,
        "health": health_state,
        "predict": {
            "url": predict_url,
            "request_count": len(sample_passwords),
            "response_count": len(sample_passwords),
            "rounds": rounds,
            "checked_rows": len(sample_passwords),
            "max_abs_error": max_abs_error,
            "tolerance": tolerance,
            "target_scale": "log10_times_plus_1",
            "sample_passwords": sample_passwords[:5],
            "checks": predict_checks,
        },
    }


def call_reload_model_endpoint(
    registration_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Notify the serving application to reload the production model.

    The endpoint URL and shared secret are read from environment variables so
    training artifacts and logs never need to contain the secret value.
    """
    reload_url = os.getenv("SERVICE_RELOAD_URL")
    reload_secret = os.getenv("SERVICE_RELOAD_SECRET")

    if not reload_url:
        message = "SERVICE_RELOAD_URL is not configured; skipping model reload."
        if os.getenv("CI"):
            raise RuntimeError(
                "SERVICE_RELOAD_URL is required in CI after successful training."
            )

        logger.warning(message)
        return {"status": "skipped", "reason": "SERVICE_RELOAD_URL is not configured"}

    if not reload_secret:
        if os.getenv("CI"):
            raise RuntimeError(
                "SERVICE_RELOAD_SECRET is required in CI to call /reload_model "
                "after successful training."
            )

        raise RuntimeError(
            "SERVICE_RELOAD_SECRET is required to call /reload_model when "
            "SERVICE_RELOAD_URL is configured."
        )

    headers = {"X-Service-Token": reload_secret}
    payload: dict[str, Any] | None = None
    if registration_result is not None:
        payload = {
            "model_name": registration_result.get("model_name"),
            "model_alias": registration_result.get("model_alias"),
            "expected_model_version": registration_result.get("model_version"),
        }
        payload = {key: value for key, value in payload.items() if value is not None}

    last_error: str | None = None

    for attempt in range(1, _RELOAD_ATTEMPTS + 1):
        try:
            response = requests.post(
                reload_url,
                headers=headers,
                json=payload,
                timeout=_RELOAD_TIMEOUT_SECONDS,
            )
            response.raise_for_status()

            try:
                return response.json()
            except ValueError:
                return {
                    "status": "success",
                    "http_status": response.status_code,
                }
        except requests.RequestException as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            sanitized_error = _sanitize_for_log(str(exc))
            if status_code is None:
                last_error = type(exc).__name__
                if sanitized_error:
                    last_error = f"{last_error}: {sanitized_error}"
            else:
                last_error = f"{type(exc).__name__} with HTTP status {status_code}"

            if attempt < _RELOAD_ATTEMPTS:
                time.sleep(_RELOAD_RETRY_DELAY_SECONDS)

    raise RuntimeError(
        "Failed to reload service model after "
        f"{_RELOAD_ATTEMPTS} attempts. Last error: {_sanitize_for_log(last_error)}."
    )


def _read_validated_training_dataframe(data_path: str):
    """Read and clean training data after the file-level validation passed."""
    raw_df = read_csv(data_path)
    is_valid, errors, cleaned_df, _metrics = validate_password_dataframe(raw_df)
    if not is_valid or cleaned_df is None:
        raise RuntimeError(
            "Data validation failed after validation gate passed: " + "; ".join(errors)
        )

    return cleaned_df


def _validation_report_dict(validation_result) -> dict[str, Any]:
    schema_metrics = getattr(validation_result, "metrics", None)
    thresholds = getattr(validation_result, "thresholds", None)
    if thresholds is None and isinstance(schema_metrics, dict):
        thresholds = schema_metrics.get("thresholds")

    report = {
        "is_valid": validation_result.is_valid,
        "errors": validation_result.errors,
        "n_rows": validation_result.n_rows,
        "columns": validation_result.columns,
    }
    if hasattr(validation_result, "metrics"):
        report["schema_metrics"] = schema_metrics
    if thresholds is not None:
        report["thresholds"] = thresholds
    return report


def _merged_validation_report_dict(
    schema_result, model_quality_result
) -> dict[str, Any]:
    schema_metrics = getattr(schema_result, "metrics", None)
    thresholds = getattr(schema_result, "thresholds", None)
    if thresholds is None and isinstance(schema_metrics, dict):
        thresholds = schema_metrics.get("thresholds")

    model_quality_errors = list(getattr(model_quality_result, "errors", []))
    report = {
        "is_valid": schema_result.is_valid,
        "errors": list(schema_result.errors),
        "n_rows": schema_result.n_rows,
        "columns": schema_result.columns,
        "schema_metrics": schema_metrics,
        "model_quality_is_valid": model_quality_result.is_valid,
        "model_quality_errors": model_quality_errors,
        "model_quality_metrics": model_quality_result.metrics,
    }
    if thresholds is not None:
        report["thresholds"] = thresholds
    return report


def _write_validation_report(report: dict[str, Any], report_path: str) -> None:
    output_file = Path(report_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_evidently_warning_report(
    error: Exception, output_path: str
) -> dict[str, Any]:
    report = {
        "status": "warning",
        "warning": f"Evidently tests failed to run: {error}",
        "evidently_failed": True,
    }
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _run_non_blocking_model_quality_validation(df):
    """Evaluate old-prod model quality without making it a pipeline gate."""
    try:
        return validate_model_quality_with_prod_model(df)
    except Exception as exc:
        logger.warning(
            "Model-quality validation failed to run after data quality checks; "
            "continuing pipeline: %s",
            _sanitize_for_log(str(exc)),
        )
        return SimpleNamespace(
            is_valid=False,
            errors=[f"Model-quality validation failed to run: {exc}"],
            metrics={},
            scored_df=None,
        )


def _run_non_blocking_evidently_tests(
    df, output_path: str = _DEFAULT_EVIDENTLY_REPORT_PATH
) -> dict[str, Any]:
    """Run Evidently after validation gates without making its status blocking."""
    try:
        return run_evidently_tests(df, output_path)
    except Exception as exc:
        logger.warning(
            "Evidently tests failed after validation gates; continuing pipeline: %s",
            _sanitize_for_log(str(exc)),
        )
        return _write_evidently_warning_report(exc, output_path)


def run_training_pipeline(data_url: str | None = None) -> dict[str, Any]:
    """Run the training orchestration from download through service reload.

    Reload is intentionally called only after these gates complete successfully:
    data download, data validation, model training, MLflow registration, and
    explicit read-back verification that the production alias points to the new
    model version.
    """
    pipeline_started_at = time.monotonic()
    try:
        resolved_data_url = data_url or os.getenv("DATA_URL")
        if not resolved_data_url:
            raise ValueError("DATA_URL environment variable is required.")

        Path("artifacts").mkdir(parents=True, exist_ok=True)
        Path("reports").mkdir(parents=True, exist_ok=True)
        Path("validation_reports").mkdir(parents=True, exist_ok=True)

        data_path = _timed_step(
            "data download", download_data, resolved_data_url, _DEFAULT_DATA_PATH
        )
        logger.info("data downloaded: %s", _sanitize_for_log(data_path))
        schema_result = _timed_step(
            "schema validation",
            validate_data_file,
            data_path,
            _DEFAULT_VALIDATION_REPORT_PATH,
        )
        validation_report = _validation_report_dict(schema_result)
        if not schema_result.is_valid:
            errors = schema_result.errors
            error_message = "; ".join(errors) or "unknown validation error"
            logger.info("validation failed: %s", _sanitize_for_log(error_message))
            logger.warning(
                "VALIDATION_FAILED_NO_MODEL_REGISTERED: errors=%s metrics=%s",
                _sanitize_for_log(errors),
                _sanitize_for_log(validation_report.get("schema_metrics")),
            )
            if _env_flag_is_true("FAIL_CI_ON_VALIDATION_FAILED"):
                raise SystemExit(2)

            return {
                "status": "validation_failed",
                "data_path": data_path,
                "validation_report": validation_report,
                "errors": errors,
            }

        logger.info("validation passed")

        training_df: pd.DataFrame
        if hasattr(schema_result, "cleaned_df"):
            training_df = schema_result.cleaned_df
            if training_df is None:
                raise RuntimeError(
                    "Data validation succeeded but cleaned data is missing; "
                    "this indicates an internal validation error."
                )
        else:
            training_df = _read_validated_training_dataframe(data_path)

        model, metrics = _timed_step(
            "training", train_password_model, training_df[["Password", "Times"]]
        )
        logger.info("model trained")

        model_quality_result = _timed_step(
            "model-quality validation",
            _run_non_blocking_model_quality_validation,
            training_df,
        )
        validation_report = _merged_validation_report_dict(
            schema_result, model_quality_result
        )
        _write_validation_report(validation_report, _DEFAULT_VALIDATION_REPORT_PATH)

        if not model_quality_result.is_valid:
            model_quality_errors = list(getattr(model_quality_result, "errors", []))
            error_message = (
                "; ".join(model_quality_errors)
                or "unknown model quality validation error"
            )
            logger.warning(
                "model quality validation failed after data quality checks passed; "
                "continuing to train and register new model version: %s",
                _sanitize_for_log(error_message),
            )

        evidently_input_df: pd.DataFrame | None = getattr(
            model_quality_result, "scored_df", None
        )
        if evidently_input_df is None:
            evidently_input_df = training_df
        evidently_report = _timed_step(
            "Evidently",
            _run_non_blocking_evidently_tests,
            evidently_input_df,
            _DEFAULT_EVIDENTLY_REPORT_PATH,
        )

        registration_result = _timed_step(
            "MLflow registration",
            register_model_in_mlflow,
            model,
            metrics,
            validation_report=validation_report,
            model_alias=_PRODUCTION_MODEL_ALIAS,
        )
        logger.info("model registered: %s", _sanitize_for_log(registration_result))
        _timed_step(
            "alias verification",
            _ensure_registration_alias_verified,
            registration_result,
        )
        logger.info("model alias verified: %s", _sanitize_for_log(registration_result))

        model_artifact_path = _timed_step(
            "model artifact saving",
            save_model_artifact,
            model,
            _DEFAULT_MODEL_ARTIFACT_PATH,
        )
        reload_result = _timed_step(
            "service reload", call_reload_model_endpoint, registration_result
        )
        logger.info(
            "service reload response received: %s", _sanitize_for_log(reload_result)
        )
        try:
            serving_verification = _timed_step(
                "serving verification",
                verify_serving_after_reload,
                registration_result,
                reload_result,
                training_df,
            )
            pipeline_status = "success"
        except Exception as exc:
            if _env_flag_is_true(_FAIL_CI_ON_SERVING_VERIFICATION_FAILED_ENV):
                raise

            serving_verification = {
                "status": "warning",
                "reason": "serving verification failed after reload",
                "error_type": type(exc).__name__,
                "error": str(_sanitize_for_log(str(exc))),
            }
            pipeline_status = "success_with_serving_verification_warning"
            logger.warning(
                "serving verification failed after service reload; continuing "
                "because %s is not enabled: %s",
                _FAIL_CI_ON_SERVING_VERIFICATION_FAILED_ENV,
                _sanitize_for_log(str(exc)),
            )
        logger.info(
            "service state checked: %s", _sanitize_for_log(serving_verification)
        )
        logger.info(
            "service loaded model version: %s",
            _sanitize_for_log(serving_verification.get("loaded_model_version")),
        )
        return {
            "status": pipeline_status,
            "data_path": data_path,
            "evidently_report": evidently_report,
            "validation_report": validation_report,
            "model_artifact_path": model_artifact_path,
            "registration": registration_result,
            "reload": reload_result,
            "serving_verification": serving_verification,
        }
    finally:
        logger.info(
            "%s took %.3f sec",
            "total pipeline",
            time.monotonic() - pipeline_started_at,
        )


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    result = run_training_pipeline()
    logger.info("Training pipeline completed: %s", _sanitize_for_log(result))


if __name__ == "__main__":
    main()
