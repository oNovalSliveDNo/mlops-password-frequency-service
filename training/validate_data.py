from __future__ import annotations

import numpy as np
import pandas as pd


def validate_password_dataframe(
    df: pd.DataFrame,
) -> tuple[bool, list[str], pd.DataFrame | None]:
    """Validate and clean password-frequency training data.

    The expected input contains password strings in ``Password`` and positive
    numeric frequencies in ``Times``. On success, a cleaned dataframe with only
    these two columns is returned.
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

    finite_times = np.isfinite(
        cleaned_df["Times"].to_numpy(dtype=float, na_value=np.nan)
    )
    if not finite_times.all():
        errors.append("Column Times contains infinite values")

    positive_times = cleaned_df["Times"] > 0
    if not positive_times.all():
        errors.append("Column Times must contain only positive values")

    if errors:
        return False, errors, None

    return True, [], cleaned_df
