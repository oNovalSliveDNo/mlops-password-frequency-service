import logging
import os
import time
from pathlib import Path
from typing import Any

import requests


_RELOAD_ATTEMPTS = 3
_RELOAD_RETRY_DELAY_SECONDS = 2
_RELOAD_TIMEOUT_SECONDS = 10
_DEFAULT_DATA_PATH = "artifacts/new_data.csv"
_DEFAULT_MODEL_ARTIFACT_PATH = "artifacts/model.joblib"
_DEFAULT_VALIDATION_REPORT_PATH = "validation_reports/validation_report.json"
_DEFAULT_EVIDENTLY_REPORT_PATH = "reports/tests.json"
_PRODUCTION_MODEL_ALIAS = "prod"


logger = logging.getLogger(__name__)


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


def train_password_model(df):
    from training.train_model import train_password_model as train_password_model_impl

    return train_password_model_impl(df)


def save_model_artifact(model, output_path: str = _DEFAULT_MODEL_ARTIFACT_PATH) -> str:
    from training.train_model import save_model_artifact as save_model_artifact_impl

    return save_model_artifact_impl(model, output_path)


def register_model_in_mlflow(
    model, metrics: dict, validation_report: dict | None = None
) -> dict:
    from training.register_model import register_model_in_mlflow as register_model_impl

    return register_model_impl(model, metrics, validation_report=validation_report)


def call_reload_model_endpoint() -> dict[str, Any]:
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

    headers = {"X-Service-Token": reload_secret or ""}
    last_error: str | None = None

    for attempt in range(1, _RELOAD_ATTEMPTS + 1):
        try:
            response = requests.post(
                reload_url,
                headers=headers,
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
            if status_code is None:
                last_error = type(exc).__name__
            else:
                last_error = f"{type(exc).__name__} with HTTP status {status_code}"

            if attempt < _RELOAD_ATTEMPTS:
                time.sleep(_RELOAD_RETRY_DELAY_SECONDS)

    raise RuntimeError(
        "Failed to reload service model after "
        f"{_RELOAD_ATTEMPTS} attempts. Last error: {last_error}."
    )


def _read_validated_training_dataframe(data_path: str):
    """Read and clean training data after the file-level validation passed."""
    raw_df = read_csv(data_path)
    is_valid, errors, cleaned_df = validate_password_dataframe(raw_df)
    if not is_valid or cleaned_df is None:
        raise RuntimeError(
            "Data validation failed after validation gate passed: " + "; ".join(errors)
        )

    return cleaned_df


def _validation_report_dict(validation_result) -> dict[str, Any]:
    return {
        "is_valid": validation_result.is_valid,
        "errors": validation_result.errors,
        "n_rows": validation_result.n_rows,
        "columns": validation_result.columns,
    }


def run_training_pipeline(data_url: str | None = None) -> dict[str, Any]:
    """Run the training orchestration from download through service reload.

    Reload is intentionally called only after these gates complete successfully:
    data download, data validation, model training, and MLflow registration with
    the production alias set by ``register_model_in_mlflow``.
    """
    resolved_data_url = data_url or os.getenv("DATA_URL")
    if not resolved_data_url:
        raise ValueError("DATA_URL environment variable is required.")

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("validation_reports").mkdir(parents=True, exist_ok=True)

    data_path = download_data(resolved_data_url, _DEFAULT_DATA_PATH)
    validation_result = validate_data_file(data_path, _DEFAULT_VALIDATION_REPORT_PATH)
    validation_report = _validation_report_dict(validation_result)
    if not validation_result.is_valid:
        errors = validation_result.errors
        error_message = "; ".join(errors) or "unknown validation error"
        logger.error("Data validation failed: %s", error_message)
        return {
            "status": "validation_failed",
            "data_path": data_path,
            "validation_report": validation_report,
            "errors": errors,
        }

    if hasattr(validation_result, "cleaned_df"):
        training_df = validation_result.cleaned_df
        if training_df is None:
            raise RuntimeError(
                "Data validation succeeded but cleaned data is missing; "
                "this indicates an internal validation error."
            )
    else:
        # Backward-compatible path for tests or external callers that provide
        # ValidationResult-like objects without the new cleaned_df field.
        training_df = _read_validated_training_dataframe(data_path)

    evidently_report = run_evidently_tests(training_df, _DEFAULT_EVIDENTLY_REPORT_PATH)
    model, metrics = train_password_model(training_df)
    model_artifact_path = save_model_artifact(model, _DEFAULT_MODEL_ARTIFACT_PATH)

    registration_result = register_model_in_mlflow(
        model,
        metrics,
        validation_report=validation_report,
    )

    reload_result = {"status": "skipped", "reason": "MODEL_ALIAS is not prod"}
    if registration_result.get("model_alias") == _PRODUCTION_MODEL_ALIAS:
        reload_result = call_reload_model_endpoint()

    return {
        "data_path": data_path,
        "evidently_report": evidently_report,
        "validation_report": validation_report,
        "model_artifact_path": model_artifact_path,
        "registration": registration_result,
        "reload": reload_result,
    }


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    result = run_training_pipeline()
    logger.info("Training pipeline completed: %s", result)


if __name__ == "__main__":
    main()
