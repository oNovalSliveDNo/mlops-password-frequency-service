import json

import pandas as pd
import pytest

from training.validate_data import validate_password_dataframe


def make_lowercase_passwords(n: int) -> list[str]:
    passwords = []

    for first in "abcdefghijklmnopqrstuvwxyz":
        for second in "abcdefghijklmnopqrstuvwxyz":
            passwords.append(f"password{first}{second}")

            if len(passwords) == n:
                return passwords

    raise ValueError("Not enough generated passwords")


def test_validation_accepts_valid_binary_lowercase_data():
    df = pd.DataFrame({"Password": ["abcd", "efgh"], "Times": [0.0, 1.0]})

    is_valid, errors, cleaned_df, metrics = validate_password_dataframe(df)

    assert is_valid is True
    assert errors == []
    assert cleaned_df is not None
    assert list(cleaned_df.columns) == ["Password", "Times"]
    assert cleaned_df["Password"].tolist() == ["abcd", "efgh"]
    assert cleaned_df["Times"].tolist() == [0.0, 1.0]
    assert metrics == {
        "n_rows": 2,
        "columns": ["Password", "Times"],
        "password_min_length": 4,
        "password_max_length": 4,
        "password_mean_length": 4.0,
        "times_min": 0.0,
        "times_max": 1.0,
        "times_unique_values": [0.0, 1.0],
        "positive_target_share": 0.5,
        "n_duplicate_rows": 0,
    }


def test_validation_rejects_extra_columns():
    df = pd.DataFrame(
        {"Password": ["abcd", "efgh"], "Times": [0.0, 1.0], "Garbage": ["x", "y"]}
    )

    is_valid, errors, cleaned_df, metrics = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert metrics is None
    assert any(
        "exactly match" in error or "unexpected column" in error for error in errors
    )
    assert any("Garbage" in error for error in errors)


@pytest.mark.parametrize(
    "times_values",
    [
        [0.0, 0.5, 1.0],
        [0.0, 2.0],
        [-1.0, 1.0],
    ],
)
def test_validation_rejects_non_binary_times(times_values):
    df = pd.DataFrame(
        {
            "Password": make_lowercase_passwords(len(times_values)),
            "Times": times_values,
        }
    )

    is_valid, errors, cleaned_df, metrics = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert metrics is None
    assert any("{0.0, 1.0}" in error for error in errors)


@pytest.mark.parametrize(
    "password",
    [
        "ABC",
        "abc123",
        "abc def",
        "abc!",
        "",
    ],
)
def test_validation_rejects_invalid_password_characters(password):
    df = pd.DataFrame({"Password": [password, "valid"], "Times": [0.0, 1.0]})

    is_valid, errors, cleaned_df, metrics = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert metrics is None
    assert any(
        "lowercase letters a-z" in error or "empty values" in error for error in errors
    )


def test_validation_rejects_low_positive_share():
    df = pd.DataFrame(
        {
            "Password": make_lowercase_passwords(100),
            "Times": [0.0] * 80 + [1.0] * 20,
        }
    )

    is_valid, errors, cleaned_df, metrics = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert metrics is None
    assert any("invalid target distribution" in error for error in errors)


def test_validation_rejects_high_positive_share():
    df = pd.DataFrame(
        {
            "Password": make_lowercase_passwords(100),
            "Times": [0.0] * 20 + [1.0] * 80,
        }
    )

    is_valid, errors, cleaned_df, metrics = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert metrics is None
    assert any("invalid target distribution" in error for error in errors)


def test_validate_data_file_writes_report(tmp_path):
    from training.validate_data import validate_data_file

    input_path = tmp_path / "passwords.csv"
    report_path = tmp_path / "report.json"
    input_path.write_text("Password,Times\nabcd,0.0\nefgh,1.0\n", encoding="utf-8")

    result = validate_data_file(str(input_path), str(report_path))

    assert result.is_valid is True
    assert result.metrics == {
        "n_rows": 2,
        "columns": ["Password", "Times"],
        "password_min_length": 4,
        "password_max_length": 4,
        "password_mean_length": 4.0,
        "times_min": 0.0,
        "times_max": 1.0,
        "times_unique_values": [0.0, 1.0],
        "positive_target_share": 0.5,
        "n_duplicate_rows": 0,
    }

    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["is_valid"] is True
    assert report["errors"] == []
    assert report["n_rows"] == 2
    assert report["columns"] == ["Password", "Times"]
    assert report["metrics"] == result.metrics


def test_validate_data_file_writes_invalid_report(tmp_path):
    from training.validate_data import ValidationResult, validate_data_file

    input_path = tmp_path / "passwords.csv"
    report_path = tmp_path / "report.json"
    input_path.write_text("Password,Times\nvalid,not-a-number\n", encoding="utf-8")

    result = validate_data_file(str(input_path), str(report_path))

    assert result == ValidationResult(
        False, ["Column Times contains non-numeric values"], 0, []
    )
    assert report_path.read_text(encoding="utf-8") == (
        "{\n"
        '  "is_valid": false,\n'
        '  "errors": [\n'
        '    "Column Times contains non-numeric values"\n'
        "  ],\n"
        '  "n_rows": 0,\n'
        '  "columns": [],\n'
        '  "metrics": null\n'
        "}"
    )


def test_validate_data_file_writes_read_error_report(tmp_path):
    from training.validate_data import validate_data_file

    input_path = tmp_path / "missing.csv"
    report_path = tmp_path / "report.json"

    result = validate_data_file(str(input_path), str(report_path))

    assert result.is_valid is False
    assert result.n_rows == 0
    assert result.columns == []
    assert len(result.errors) == 1
    assert result.errors[0].startswith("Failed to read CSV file:")
    report_text = report_path.read_text(encoding="utf-8")
    assert '"is_valid": false' in report_text
    assert '"errors": [' in report_text
    assert '"n_rows": 0' in report_text
    assert '"columns": []' in report_text
