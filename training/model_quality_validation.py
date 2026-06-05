from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from training.validation_thresholds import DEFAULT_VALIDATION_THRESHOLDS

_DEFAULT_THRESHOLDS_PATH = "reference/validation_thresholds.json"
_MODEL_QUALITY_SECTION = "model_quality"


@dataclass
class ModelQualityValidationResult:
    is_valid: bool
    errors: list[str]
    metrics: dict[str, Any]
    scored_df: pd.DataFrame | None = None


def load_validation_thresholds(path: str = _DEFAULT_THRESHOLDS_PATH) -> dict[str, Any]:
    """Load schema and model-quality thresholds with defaults for missing keys."""
    thresholds: dict[str, Any] = deepcopy(DEFAULT_VALIDATION_THRESHOLDS)
    threshold_path = Path(path)

    if threshold_path.exists():
        loaded = json.loads(threshold_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            for section, loaded_section in loaded.items():
                default_section = thresholds.get(section)
                if isinstance(default_section, dict) and isinstance(
                    loaded_section, dict
                ):
                    thresholds[section] = {**default_section, **loaded_section}
                else:
                    thresholds[section] = loaded_section

    for section, defaults in DEFAULT_VALIDATION_THRESHOLDS.items():
        if not isinstance(thresholds.get(section), dict):
            thresholds[section] = deepcopy(defaults)
            continue

        for key, default_value in defaults.items():
            thresholds[section].setdefault(key, default_value)

    return thresholds


def get_prod_model_uri() -> str:
    model_name = os.getenv("MODEL_NAME")
    if not model_name:
        raise RuntimeError(
            "MODEL_NAME environment variable is required to load the production model."
        )

    model_alias = os.getenv("MODEL_ALIAS", "prod")
    return f"models:/{model_name}@{model_alias}"


def load_prod_model_from_mlflow():
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    return mlflow.pyfunc.load_model(get_prod_model_uri())


def normalize_predictions(predictions) -> np.ndarray:
    if hasattr(predictions, "to_numpy"):
        values = predictions.to_numpy().ravel().tolist()
    elif hasattr(predictions, "ravel"):
        values = predictions.ravel().tolist()
    elif hasattr(predictions, "tolist"):
        values = predictions.tolist()
    else:
        values = list(predictions)

    return np.asarray(values, dtype=float).ravel()


def _threshold_error(
    metric_name: str, metric_value: float, threshold_name: str, threshold_value: float
) -> str:
    return (
        f"Model quality metric {metric_name}={metric_value:.6f} exceeds "
        f"threshold {threshold_name}={threshold_value:.6f}"
    )


def _validate_thresholds(
    metrics: dict[str, Any], thresholds: dict[str, Any]
) -> list[str]:
    quality_thresholds = thresholds.get(_MODEL_QUALITY_SECTION, {})
    errors: list[str] = []

    threshold_checks = (
        ("rmse", "max_rmse"),
        ("mae", "max_mae"),
        ("abs_mean_error", "max_abs_mean_error"),
        ("prediction_target_mean_gap", "max_prediction_target_mean_gap"),
    )

    for metric_name, threshold_name in threshold_checks:
        if threshold_name not in quality_thresholds:
            continue

        metric_value = float(metrics[metric_name])
        threshold_value = float(quality_thresholds[threshold_name])
        if metric_value > threshold_value:
            errors.append(
                _threshold_error(
                    metric_name, metric_value, threshold_name, threshold_value
                )
            )

    return errors


def validate_model_quality_with_prod_model(
    df,
    thresholds_path: str = _DEFAULT_THRESHOLDS_PATH,
    prod_model=None,
) -> ModelQualityValidationResult:
    thresholds = load_validation_thresholds(thresholds_path)

    try:
        if prod_model is None:
            prod_model = load_prod_model_from_mlflow()

        predictions = prod_model.predict(df["Password"])
        prediction_values = normalize_predictions(predictions)

        row_count = int(len(df.index))
        if len(prediction_values) != row_count:
            return ModelQualityValidationResult(
                is_valid=False,
                errors=[
                    "Prediction length does not match input row count: "
                    f"{len(prediction_values)} != {row_count}"
                ],
                metrics={"row_count": row_count},
            )

        if not np.isfinite(prediction_values).all():
            return ModelQualityValidationResult(
                is_valid=False,
                errors=["Predictions contain non-finite values"],
                metrics={"row_count": row_count},
            )

        y_true = np.log10(df["Times"].astype(float).to_numpy() + 1)
        if not np.isfinite(y_true).all():
            return ModelQualityValidationResult(
                is_valid=False,
                errors=["Target log values contain non-finite values"],
                metrics={"row_count": row_count},
            )

        prediction_error = prediction_values - y_true
        rmse = float(np.sqrt(np.mean(np.square(prediction_error))))
        mae = float(np.mean(np.abs(prediction_error)))
        mean_error = float(np.mean(prediction_error))
        abs_mean_error = float(abs(mean_error))
        prediction_mean = float(np.mean(prediction_values))
        target_log_mean = float(np.mean(y_true))
        prediction_target_mean_gap = float(abs(prediction_mean - target_log_mean))
        positive_share = float((df["Times"].astype(float).to_numpy() > 0).mean())

        metrics: dict[str, Any] = {
            "rmse": rmse,
            "mae": mae,
            "mean_error": mean_error,
            "abs_mean_error": abs_mean_error,
            "prediction_mean": prediction_mean,
            "target_log_mean": target_log_mean,
            "mean_gap": prediction_target_mean_gap,
            "prediction_target_mean_gap": prediction_target_mean_gap,
            "positive_share": positive_share,
            "row_count": row_count,
            "scale_marker": "log10_times_plus_1",
        }

        scored_df = df.copy()
        scored_df["target_log"] = y_true
        scored_df["prediction"] = prediction_values
        scored_df["prediction_error"] = prediction_error

        errors = _validate_thresholds(metrics, thresholds)
        return ModelQualityValidationResult(
            is_valid=not errors,
            errors=errors,
            metrics=metrics,
            scored_df=scored_df,
        )
    except Exception as exc:
        return ModelQualityValidationResult(
            is_valid=False,
            errors=[f"Model quality validation failed: {exc}"],
            metrics={},
        )
