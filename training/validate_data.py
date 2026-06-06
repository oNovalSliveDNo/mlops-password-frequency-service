from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfDuplicatedColumns,
    TestNumberOfEmptyColumns,
    TestNumberOfEmptyRows,
    TestNumberOfRows,
    TestShareOfMissingValues,
)


from training.validation_thresholds import (
    DEFAULT_SCHEMA_THRESHOLDS,
    load_validation_thresholds,
)

_MIN_POSITIVE_TARGET_SHARE = float(
    DEFAULT_SCHEMA_THRESHOLDS["min_positive_target_share"]
)
_MAX_POSITIVE_TARGET_SHARE = float(
    DEFAULT_SCHEMA_THRESHOLDS["max_positive_target_share"]
)
_DEFAULT_MAX_PASSWORD_LENGTH = int(DEFAULT_SCHEMA_THRESHOLDS["max_password_length"])
_REQUIRED_COLUMNS = ["Password", "Times"]
_PASSWORD_PATTERN = re.compile(r"^[a-z]+$")
_ALLOWED_TIMES_VALUES = {0.0, 1.0}


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: list[str]
    n_rows: int
    columns: list[str]
    cleaned_df: pd.DataFrame | None = None
    metrics: dict[str, Any] | None = None
    thresholds: dict[str, Any] | None = None


def run_evidently_tests(df: pd.DataFrame, output_path: str = "tests.json") -> dict:
    """Run non-blocking Evidently data quality checks and save their report."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        tests = TestSuite(
            tests=[
                TestNumberOfRows(gt=0),
                TestShareOfMissingValues(eq=0),
                TestNumberOfEmptyRows(eq=0),
                TestNumberOfEmptyColumns(eq=0),
                TestNumberOfDuplicatedColumns(eq=0),
            ]
        )
        tests.run(reference_data=None, current_data=df)
        report = tests.as_dict()
        if "status" not in report:
            summary = report.get("summary", {})
            report["status"] = (
                "success" if summary.get("all_passed") is True else "failure"
            )
    except Exception as exc:
        report = {
            "status": "warning",
            "warning": f"Evidently tests failed to run: {exc}",
            "evidently_failed": True,
        }

    output_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _schema_thresholds() -> dict[str, Any]:
    return dict(load_validation_thresholds()["schema"])


def _safe_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _empty_mask(df: pd.DataFrame) -> pd.DataFrame:
    string_empty = df.apply(
        lambda column: column.map(
            lambda value: isinstance(value, str) and value.strip() == ""
        )
    )
    return df.isna() | string_empty


def _build_raw_validation_diagnostics(
    df: pd.DataFrame, thresholds: dict[str, Any]
) -> dict[str, Any]:
    """Build best-effort diagnostics from untrusted raw input data."""
    diagnostics: dict[str, Any] = {
        "n_rows": int(len(df.index)),
        "columns": [str(column) for column in df.columns],
        "n_duplicate_rows": None,
        "n_missing_rows": None,
        "n_empty_rows": None,
        "n_missing_columns": None,
        "n_empty_columns": None,
        "duplicate_columns": [
            str(column) for column in df.columns[df.columns.duplicated()].tolist()
        ],
        "password_min_length": None,
        "password_max_length": None,
        "password_mean_length": None,
        "times_min": None,
        "times_max": None,
        "times_unique_values": None,
        "positive_target_share": None,
        "thresholds": dict(thresholds),
    }

    try:
        diagnostics["n_duplicate_rows"] = int(df.duplicated().sum())
    except Exception:
        diagnostics["n_duplicate_rows"] = None

    try:
        empty_values = _empty_mask(df)
        diagnostics["n_missing_rows"] = int(df.isna().any(axis=1).sum())
        diagnostics["n_empty_rows"] = int(empty_values.all(axis=1).sum())
        diagnostics["n_missing_columns"] = int(df.isna().any(axis=0).sum())
        diagnostics["n_empty_columns"] = int(empty_values.all(axis=0).sum())
    except Exception:
        pass

    if "Password" in df.columns:
        try:
            password_series = df["Password"]
            if isinstance(password_series, pd.DataFrame):
                password_series = password_series.iloc[:, 0]
            password_lengths = password_series.dropna().astype(str).str.len()
            if not password_lengths.empty:
                diagnostics["password_min_length"] = _safe_int(password_lengths.min())
                diagnostics["password_max_length"] = _safe_int(password_lengths.max())
                diagnostics["password_mean_length"] = _safe_float(
                    password_lengths.mean()
                )
        except Exception:
            pass

    if "Times" in df.columns:
        try:
            times_series = df["Times"]
            if isinstance(times_series, pd.DataFrame):
                times_series = times_series.iloc[:, 0]
            numeric_times = pd.to_numeric(times_series, errors="coerce")
            numeric_times = numeric_times[
                np.isfinite(numeric_times.to_numpy(dtype=float))
            ]
            if not numeric_times.empty:
                diagnostics["times_min"] = _safe_float(numeric_times.min())
                diagnostics["times_max"] = _safe_float(numeric_times.max())
                diagnostics["times_unique_values"] = sorted(
                    float(value) for value in numeric_times.dropna().unique()
                )
                diagnostics["positive_target_share"] = _safe_float(
                    (numeric_times > 0).mean()
                )
        except Exception:
            pass

    return diagnostics


def _build_validation_metrics(cleaned_df: pd.DataFrame) -> dict[str, Any]:
    password_lengths = cleaned_df["Password"].str.len()
    times_values = cleaned_df["Times"]
    thresholds = _schema_thresholds()

    return {
        "n_rows": int(len(cleaned_df.index)),
        "columns": list(cleaned_df.columns),
        "password_min_length": int(password_lengths.min()),
        "password_max_length": int(password_lengths.max()),
        "password_mean_length": float(password_lengths.mean()),
        "times_min": float(times_values.min()),
        "times_max": float(times_values.max()),
        "times_unique_values": sorted(float(value) for value in times_values.unique()),
        "positive_target_share": float((times_values > 0).mean()),
        "n_duplicate_rows": int(cleaned_df.duplicated().sum()),
        "thresholds": thresholds,
    }


def validate_password_dataframe(
    df: pd.DataFrame,
    *,
    max_password_length: int | None = None,
    min_positive_target_share: float | None = None,
    max_positive_target_share: float | None = None,
) -> tuple[bool, list[str], pd.DataFrame | None, dict[str, Any] | None]:
    """Validate and clean password-frequency training data.

    The expected input contains lowercase alphabetic password strings in
    ``Password`` and binary numeric target values in ``Times``. On success, a
    cleaned dataframe with only these two columns and validation metrics are
    returned.
    """
    schema_thresholds = _schema_thresholds()
    if max_password_length is None:
        max_password_length = int(schema_thresholds["max_password_length"])
    if min_positive_target_share is None:
        min_positive_target_share = float(
            schema_thresholds["min_positive_target_share"]
        )
    if max_positive_target_share is None:
        max_positive_target_share = float(
            schema_thresholds["max_positive_target_share"]
        )

    errors: list[str] = []

    if len(df.index) == 0:
        errors.append("DataFrame must contain at least one row")

    empty_rows = df.isna().all(axis=1)
    if empty_rows.any():
        empty_row_numbers = df.index[empty_rows.to_numpy()].tolist()
        row_numbers = [str(index) for index in empty_row_numbers]
        errors.append(f"DataFrame contains fully empty rows: {', '.join(row_numbers)}")

    empty_columns = df.isna().all(axis=0)
    if empty_columns.any():
        empty_column_names = df.columns[empty_columns.to_numpy()].tolist()
        column_names = [str(column) for column in empty_column_names]
        errors.append(
            f"DataFrame contains fully empty columns: {', '.join(column_names)}"
        )

    duplicated_columns = df.columns.duplicated()
    if duplicated_columns.any():
        duplicated_column_names = df.columns[duplicated_columns].tolist()
        column_names = [str(column) for column in duplicated_column_names]
        errors.append(
            f"DataFrame contains duplicate columns: {', '.join(column_names)}"
        )

    actual_columns = list(df.columns)
    if actual_columns != _REQUIRED_COLUMNS:
        errors.append(
            "DataFrame columns must exactly match "
            f"{_REQUIRED_COLUMNS} in order; found {actual_columns}"
        )

    if errors:
        return False, errors, None, None

    cleaned_df = df.loc[:, _REQUIRED_COLUMNS].copy()

    missing_password = cleaned_df["Password"].isna()
    if missing_password.any():
        errors.append("Column Password contains missing values")

    password_values = cleaned_df["Password"].astype(str).str.strip()
    cleaned_df["Password"] = password_values

    empty_password = password_values.eq("")
    if empty_password.any():
        errors.append(
            "Column Password contains empty values after stripping whitespace"
        )

    invalid_password_pattern = ~password_values.str.match(_PASSWORD_PATTERN)
    invalid_password_pattern = (
        invalid_password_pattern & ~missing_password & ~empty_password
    )
    if invalid_password_pattern.any():
        errors.append("Column Password must contain only lowercase letters a-z")

    invalid_password_length = password_values.str.len().lt(
        1
    ) | password_values.str.len().gt(max_password_length)
    invalid_password_length = invalid_password_length & ~missing_password
    if invalid_password_length.any():
        errors.append(
            "Column Password length must be between 1 and "
            f"{max_password_length} characters"
        )

    missing_times = cleaned_df["Times"].isna()
    if missing_times.any():
        errors.append("Column Times contains missing values")

    numeric_times = pd.to_numeric(cleaned_df["Times"], errors="coerce")
    cleaned_df["Times"] = numeric_times

    invalid_times = numeric_times.isna() & ~missing_times
    if invalid_times.any():
        errors.append("Column Times contains non-numeric values")

    valid_numeric_times = numeric_times.notna()
    finite_times_mask = pd.Series(False, index=cleaned_df.index)
    if valid_numeric_times.any():
        finite_values = np.isfinite(
            numeric_times.loc[valid_numeric_times].to_numpy(dtype=float)
        )
        finite_times_mask.loc[valid_numeric_times] = finite_values
        if not finite_values.all():
            errors.append("Column Times contains infinite values")

    finite_times = numeric_times[finite_times_mask]
    invalid_allowed_values = ~finite_times.isin(_ALLOWED_TIMES_VALUES)
    if invalid_allowed_values.any():
        errors.append("Column Times values must be exactly one of {0.0, 1.0}")

    if not errors:
        positive_share = float((cleaned_df["Times"] > 0).mean())
        if not min_positive_target_share <= positive_share <= max_positive_target_share:
            errors.append(
                "Column Times has invalid target distribution: "
                f"positive share is {positive_share:.3f}; expected between "
                f"{min_positive_target_share:.2f} and {max_positive_target_share:.2f}"
            )

    if errors:
        return False, errors, None, None

    metrics = _build_validation_metrics(cleaned_df)
    return True, [], cleaned_df, metrics


def validate_data_file(
    input_path: str, report_path: str = "validation_report.json"
) -> ValidationResult:
    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        thresholds = _schema_thresholds()
        result = ValidationResult(
            False, [f"Failed to read CSV file: {exc}"], 0, [], thresholds=thresholds
        )
    else:
        thresholds = _schema_thresholds()
        raw_metrics = _build_raw_validation_diagnostics(df, thresholds)
        is_valid, errors, cleaned_df, metrics = validate_password_dataframe(df)
        if is_valid:
            if cleaned_df is None or metrics is None:
                raise RuntimeError(
                    "Validation succeeded without cleaned data or metrics"
                )
            result = ValidationResult(
                True,
                [],
                len(cleaned_df.index),
                list(cleaned_df.columns),
                cleaned_df,
                metrics,
                thresholds,
            )
        else:
            result = ValidationResult(
                False,
                errors,
                int(len(df.index)),
                [str(column) for column in df.columns],
                metrics=raw_metrics,
                thresholds=thresholds,
            )

    report = {
        "is_valid": result.is_valid,
        "errors": result.errors,
        "n_rows": result.n_rows,
        "columns": result.columns,
        "metrics": result.metrics,
        "schema_metrics": result.metrics,
        "thresholds": result.thresholds,
    }
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result
