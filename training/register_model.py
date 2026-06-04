import json
import os
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


_SENSITIVE_KEY_PARTS = ("secret", "credential", "password", "token", "key")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_sensitive_key(key: str) -> bool:
    normalized_key = key.lower()
    return any(part in normalized_key for part in _SENSITIVE_KEY_PARTS)


def _get_registered_model_version(log_model_result: Any) -> str | None:
    """Extract a registered model version from MLflow log_model results.

    MLflow versions expose slightly different result objects, so check the
    documented field first and then a few common nested/alternate locations.
    """
    for attribute in (
        "registered_model_version",
        "model_version",
        "registered_model_version_id",
        "version",
    ):
        version = getattr(log_model_result, attribute, None)
        if version:
            return str(version)

    for attribute in ("registered_model_details", "registered_model", "model"):
        nested_result = getattr(log_model_result, attribute, None)
        if nested_result is None:
            continue

        version = getattr(nested_result, "version", None) or getattr(
            nested_result, "registered_model_version", None
        )
        if version:
            return str(version)

    return None


def _find_registered_model_version(
    client: MlflowClient, model_name: str, run_id: str
) -> str | None:
    for model_version in client.search_model_versions(f"name='{model_name}'"):
        if model_version.run_id == run_id:
            return str(model_version.version)

    return None


def register_model_in_mlflow(
    model,
    metrics: dict,
    validation_report: dict | None = None,
    model_alias: str | None = None,
) -> dict:
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_name = os.getenv("MODEL_NAME")
    # Guarantee a string alias using an explicit check to satisfy type narrowing.
    resolved_model_alias: str = (
        model_alias if model_alias else os.getenv("MODEL_ALIAS", "prod")
    )
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "mlops-password-frequency-service"
    )

    if not mlflow_tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI environment variable is required.")
    if not model_name:
        raise ValueError("MODEL_NAME environment variable is required.")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        params = {}
        numeric_metrics = {}
        n_rows = None

        for key, value in metrics.items():
            if _is_sensitive_key(str(key)):
                continue

            if key == "n_rows":
                n_rows = value
            elif isinstance(value, bool) or not isinstance(value, int | float):
                params[key] = value
            else:
                numeric_metrics[key] = float(value)

        if params:
            mlflow.log_params(params)
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)
        if n_rows is not None:
            mlflow.log_param("n_rows", n_rows)

        if validation_report is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                validation_report_path = Path(temp_dir) / "validation_report.json"
                validation_report_path.write_text(
                    json.dumps(validation_report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                mlflow.log_artifact(str(validation_report_path))

        tests_report_path = _PROJECT_ROOT / "reports" / "tests.json"
        if not tests_report_path.exists():
            tests_report_path = _PROJECT_ROOT / "tests.json"
        if tests_report_path.exists():
            mlflow.log_artifact(str(tests_report_path))

        log_model_result = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        run_id = run.info.run_id

    client = MlflowClient()
    version = _get_registered_model_version(log_model_result)
    if version is None:
        version = _find_registered_model_version(client, model_name, run_id)

    if version is None:
        raise RuntimeError(
            "Could not determine registered MLflow model version for this run."
        )

    client.set_registered_model_alias(model_name, resolved_model_alias, version)

    return {
        "model_name": model_name,
        "model_alias": resolved_model_alias,
        "model_version": str(version),
        "run_id": run_id,
    }
