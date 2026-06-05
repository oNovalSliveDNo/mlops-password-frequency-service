import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

import requests

_RELOAD_ATTEMPTS = 3
_RELOAD_RETRY_DELAY_SECONDS = 10
_RELOAD_TIMEOUT_SECONDS = 120
_SERVING_CHECK_TIMEOUT_SECONDS = 30
_DEFAULT_PREDICT_SMOKE_PASSWORDS = ("password", "correcthorsebatterystaple")
_DEFAULT_DATA_PATH = "artifacts/new_data.csv"
_DEFAULT_MODEL_ARTIFACT_PATH = "artifacts/model.joblib"
_DEFAULT_VALIDATION_REPORT_PATH = "validation_reports/validation_report.json"
_DEFAULT_EVIDENTLY_REPORT_PATH = "reports/tests.json"
_PRODUCTION_MODEL_ALIAS = "prod"


logger = logging.getLogger(__name__)

_SENSITIVE_LOG_KEY_PARTS = ("secret", "credential", "password", "token", "key")
_REDACTED_LOG_VALUE = "[REDACTED]"


def _is_sensitive_log_key(key: str) -> bool:
    normalized_key = key.lower()
    return any(part in normalized_key for part in _SENSITIVE_LOG_KEY_PARTS)


def _sanitize_for_log(value: Any) -> Any:
    """Return a log-safe copy of a value by redacting known sensitive data."""
    service_reload_secret = os.getenv("SERVICE_RELOAD_SECRET")

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

    if isinstance(value, str) and service_reload_secret:
        return value.replace(service_reload_secret, _REDACTED_LOG_VALUE)

    return value


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


def _expected_model_version(registration_result: dict[str, Any] | None) -> str | None:
    if registration_result is None:
        return None

    model_version = registration_result.get("model_version")
    if model_version is None:
        return None

    return str(model_version)


def _parse_predict_smoke_passwords() -> list[str]:
    raw_passwords = os.getenv("SERVICE_PREDICT_SMOKE_PASSWORDS")
    if raw_passwords is None:
        return list(_DEFAULT_PREDICT_SMOKE_PASSWORDS)

    passwords = [password.strip() for password in raw_passwords.split(",")]
    return [password for password in passwords if password]


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
        f"Serving check {method} {url} failed after {_RELOAD_ATTEMPTS} attempts. "
        f"Last error: {_sanitize_for_log(last_error)}."
    )


def verify_serving_after_reload(
    registration_result: dict[str, Any] | None,
    reload_result: dict[str, Any],
) -> dict[str, Any]:
    """Verify that the serving app exposes the reloaded model and predicts."""
    if reload_result.get("status") == "skipped":
        return {
            "status": "skipped",
            "reason": "service reload was skipped",
        }

    expected_version = _expected_model_version(registration_result)
    health_url = _service_endpoint_url("SERVICE_HEALTH_URL", "/health")
    predict_url = _service_endpoint_url("SERVICE_PREDICT_URL", "/predict")
    if not health_url or not predict_url:
        raise RuntimeError(
            "SERVICE_HEALTH_URL and SERVICE_PREDICT_URL are required for serving "
            "verification when SERVICE_RELOAD_URL cannot be used to derive them."
        )

    health_state = _request_json_with_retries("GET", health_url)
    if health_state.get("model_loaded") is not True:
        raise RuntimeError(
            f"Serving health check reports model_loaded={health_state.get('model_loaded')!r}."
        )
    if health_state.get("last_reload_status") != "success":
        raise RuntimeError(
            "Serving health check reports last_reload_status="
            f"{health_state.get('last_reload_status')!r}."
        )
    if (
        expected_version is not None
        and health_state.get("loaded_version") != expected_version
    ):
        raise RuntimeError(
            "Serving health check loaded unexpected model version: "
            f"{health_state.get('loaded_version')!r} != {expected_version!r}."
        )

    smoke_passwords = _parse_predict_smoke_passwords()
    if not smoke_passwords:
        raise RuntimeError(
            "SERVICE_PREDICT_SMOKE_PASSWORDS did not contain any passwords."
        )

    predict_result = _request_json_with_retries(
        "POST",
        predict_url,
        json={"Password": smoke_passwords},
    )
    predictions = predict_result.get("Times")
    if not isinstance(predictions, list) or len(predictions) != len(smoke_passwords):
        raise RuntimeError(
            "Serving predict smoke check returned invalid Times: "
            f"{_sanitize_for_log(predictions)!r}."
        )

    return {
        "status": "verified",
        "expected_model_version": expected_version,
        "loaded_model_version": health_state.get("loaded_version"),
        "health": health_state,
        "predict": {
            "url": predict_url,
            "request_count": len(smoke_passwords),
            "response_count": len(predictions),
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
    report = {
        "is_valid": validation_result.is_valid,
        "errors": validation_result.errors,
        "n_rows": validation_result.n_rows,
        "columns": validation_result.columns,
    }
    if hasattr(validation_result, "metrics"):
        report["schema_metrics"] = validation_result.metrics
    return report


def _merged_validation_report_dict(
    schema_result, model_quality_result
) -> dict[str, Any]:
    return {
        "is_valid": schema_result.is_valid and model_quality_result.is_valid,
        "errors": [*schema_result.errors, *model_quality_result.errors],
        "n_rows": schema_result.n_rows,
        "columns": schema_result.columns,
        "schema_metrics": getattr(schema_result, "metrics", None),
        "model_quality_metrics": model_quality_result.metrics,
    }


def _write_validation_report(report: dict[str, Any], report_path: str) -> None:
    output_file = Path(report_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_training_pipeline(data_url: str | None = None) -> dict[str, Any]:
    """Run the training orchestration from download through service reload.

    Reload is intentionally called only after these gates complete successfully:
    data download, data validation, model training, MLflow registration, and
    explicit read-back verification that the production alias points to the new
    model version.
    """
    resolved_data_url = data_url or os.getenv("DATA_URL")
    if not resolved_data_url:
        raise ValueError("DATA_URL environment variable is required.")

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("validation_reports").mkdir(parents=True, exist_ok=True)

    data_path = download_data(resolved_data_url, _DEFAULT_DATA_PATH)
    logger.info("data downloaded: %s", _sanitize_for_log(data_path))
    schema_result = validate_data_file(data_path, _DEFAULT_VALIDATION_REPORT_PATH)
    validation_report = _validation_report_dict(schema_result)
    if not schema_result.is_valid:
        errors = schema_result.errors
        error_message = "; ".join(errors) or "unknown validation error"
        logger.info("validation failed: %s", _sanitize_for_log(error_message))
        return {
            "status": "validation_failed",
            "data_path": data_path,
            "validation_report": validation_report,
            "errors": errors,
        }

    logger.info("validation passed")

    if hasattr(schema_result, "cleaned_df"):
        training_df = schema_result.cleaned_df
        if training_df is None:
            raise RuntimeError(
                "Data validation succeeded but cleaned data is missing; "
                "this indicates an internal validation error."
            )
    else:
        # Backward-compatible path for tests or external callers that provide
        # ValidationResult-like objects without the new cleaned_df field.
        training_df = _read_validated_training_dataframe(data_path)

    model_quality_result = validate_model_quality_with_prod_model(training_df)
    validation_report = _merged_validation_report_dict(
        schema_result, model_quality_result
    )
    _write_validation_report(validation_report, _DEFAULT_VALIDATION_REPORT_PATH)

    if not model_quality_result.is_valid:
        errors = validation_report["errors"]
        error_message = "; ".join(errors) or "unknown model quality validation error"
        logger.info(
            "model quality validation failed: %s", _sanitize_for_log(error_message)
        )
        return {
            "status": "validation_failed",
            "data_path": data_path,
            "validation_report": validation_report,
            "errors": errors,
        }

    evidently_input_df = (
        model_quality_result.scored_df
        if model_quality_result.scored_df is not None
        else training_df
    )
    evidently_report = run_evidently_tests(
        evidently_input_df, _DEFAULT_EVIDENTLY_REPORT_PATH
    )
    model, metrics = train_password_model(training_df[["Password", "Times"]])
    logger.info("model trained")
    model_artifact_path = save_model_artifact(model, _DEFAULT_MODEL_ARTIFACT_PATH)

    registration_result = register_model_in_mlflow(
        model,
        metrics,
        validation_report=validation_report,
        model_alias=_PRODUCTION_MODEL_ALIAS,
    )
    logger.info("model registered: %s", _sanitize_for_log(registration_result))
    _ensure_registration_alias_verified(registration_result)
    logger.info("model alias verified: %s", _sanitize_for_log(registration_result))
    reload_result = call_reload_model_endpoint(registration_result)
    logger.info(
        "service reload response received: %s", _sanitize_for_log(reload_result)
    )
    serving_verification = verify_serving_after_reload(
        registration_result, reload_result
    )
    logger.info("service state checked: %s", _sanitize_for_log(serving_verification))
    logger.info(
        "service loaded model version: %s",
        _sanitize_for_log(serving_verification.get("loaded_model_version")),
    )

    return {
        "data_path": data_path,
        "evidently_report": evidently_report,
        "validation_report": validation_report,
        "model_artifact_path": model_artifact_path,
        "registration": registration_result,
        "reload": reload_result,
        "serving_verification": serving_verification,
    }


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    result = run_training_pipeline()
    logger.info("Training pipeline completed: %s", _sanitize_for_log(result))


if __name__ == "__main__":
    main()
