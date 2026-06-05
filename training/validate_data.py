from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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

_MIN_DISTRIBUTION_CHECK_ROWS = 5
_MIN_POSITIVE_TARGET_SHARE = 0.30


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: list[str]
    n_rows: int
    columns: list[str]
    cleaned_df: pd.DataFrame | None = None


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


def validate_password_dataframe(
    df: pd.DataFrame,
) -> tuple[bool, list[str], pd.DataFrame | None]:
    """Validate and clean password-frequency training data.

    The expected input contains password strings in ``Password`` and
    non-negative numeric target values in ``Times``. On success, a cleaned
    dataframe with only these two columns is returned.
    """
    errors: list[str] = []

    if df.empty or len(df.index) == 0:
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

    required_columns = ["Password", "Times"]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        errors.append(
            f"DataFrame is missing required columns: {', '.join(missing_columns)}"
        )

    if errors:
        return False, errors, None

    cleaned_df = df.loc[:, required_columns].copy()

    missing_password = cleaned_df["Password"].isna()
    if missing_password.any():
        errors.append("Column Password contains missing values")

    missing_times = cleaned_df["Times"].isna()
    if missing_times.any():
        errors.append("Column Times contains missing values")

    cleaned_df["Password"] = cleaned_df["Password"].astype(str).str.strip()
    empty_password = cleaned_df["Password"].eq("")
    if empty_password.any():
        errors.append(
            "Column Password contains empty values after stripping whitespace"
        )

    cleaned_df["Times"] = pd.to_numeric(cleaned_df["Times"], errors="coerce")
    invalid_times = cleaned_df["Times"].isna()
    if invalid_times.any():
        errors.append("Column Times contains non-numeric values")

    valid_times_mask = ~cleaned_df["Times"].isna()
    if valid_times_mask.any():
        finite_times = np.isfinite(
            cleaned_df.loc[valid_times_mask, "Times"].to_numpy(dtype=float)
        )
        if not finite_times.all():
            errors.append("Column Times contains infinite values")

        non_negative_times = cleaned_df.loc[valid_times_mask, "Times"] >= 0
        if not non_negative_times.all():
            errors.append("Column Times must contain only non-negative values")

    if not errors and len(cleaned_df.index) >= _MIN_DISTRIBUTION_CHECK_ROWS:
        positive_share = float((cleaned_df["Times"] > 0).mean())

        if positive_share < _MIN_POSITIVE_TARGET_SHARE:
            errors.append(
                "Column Times has invalid target distribution: "
                f"positive share is {positive_share:.3f}; expected at least "
                f"{_MIN_POSITIVE_TARGET_SHARE:.2f}"
            )

    if errors:
        return False, errors, None

    return True, [], cleaned_df


def validate_data_file(
    input_path: str, report_path: str = "validation_report.json"
) -> ValidationResult:
    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        result = ValidationResult(False, [f"Failed to read CSV file: {exc}"], 0, [])
    else:
        is_valid, errors, cleaned_df = validate_password_dataframe(df)
        if is_valid:
            if cleaned_df is None:
                raise RuntimeError("Validation succeeded without cleaned data")
            result = ValidationResult(
                True, [], len(cleaned_df.index), list(cleaned_df.columns), cleaned_df
            )
        else:
            result = ValidationResult(False, errors, 0, [])

    report = {
        "is_valid": result.is_valid,
        "errors": result.errors,
        "n_rows": result.n_rows,
        "columns": result.columns,
    }
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result
