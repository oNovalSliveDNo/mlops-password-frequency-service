import json


import numpy as np
import pandas as pd

from training.validate_data import validate_password_dataframe


def make_lowercase_passwords(n: int) -> list[str]:
    passwords = []

    for first in "abcdefghijklmnopqrstuvwxyz":
        for second in "abcdefghijklmnopqrstuvwxyz":
            passwords.append(f"password{first}{second}")

            if len(passwords) == n:
                return passwords

    raise ValueError("Not enough generated passwords")


def test_good_dataframe_is_valid():
    df = pd.DataFrame({"Password": ["qwerty", "abcde"], "Times": [0.0, 1.0]})

    is_valid, errors, cleaned_df, metrics = validate_password_dataframe(df)

    assert is_valid is True
    assert errors == []
    assert cleaned_df is not None
    assert metrics is not None
    assert metrics["n_rows"] == 2
    assert metrics["columns"] == ["Password", "Times"]
    assert metrics["password_min_length"] == 5
    assert metrics["password_max_length"] == 6
    assert metrics["password_mean_length"] == 5.5
    assert metrics["times_min"] == 0.0
    assert metrics["times_max"] == 1.0
    assert metrics["times_unique_values"] == [0.0, 1.0]
    assert metrics["positive_target_share"] == 0.5
    assert metrics["n_duplicate_rows"] == 0


def test_missing_password_column_is_invalid():
    df = pd.DataFrame({"Times": [1.0]})

    is_valid, errors, cleaned_df, metrics = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert any("Password" in error for error in errors)


def test_missing_times_column_is_invalid():
    df = pd.DataFrame({"Password": ["qwerty"]})

    is_valid, errors, cleaned_df, metrics = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert any("Times" in error for error in errors)


def test_empty_password_is_invalid():
    df = pd.DataFrame({"Password": ["   "], "Times": [1.0]})

    is_valid, errors, cleaned_df, metrics = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert any("empty values" in error for error in errors)


def test_zero_times_distribution_is_invalid():
    df = pd.DataFrame({"Password": ["qwerty"], "Times": [0]})

    is_valid, errors, cleaned_df, metrics = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert metrics is None
    assert any("invalid target distribution" in error for error in errors)


def test_negative_times_is_invalid():
    df = pd.DataFrame({"Password": ["qwerty"], "Times": [-1]})

    is_valid, errors, cleaned_df, metrics = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert any("{0.0, 1.0}" in error for error in errors)


def test_non_numeric_times_is_invalid():
    df = pd.DataFrame({"Password": ["valid"], "Times": ["not-a-number"]})

    is_valid, errors, cleaned_df, metrics = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert any("non-numeric" in error for error in errors)


def test_validate_data_file_writes_report(tmp_path):
    from training.validate_data import validate_data_file

    input_path = tmp_path / "passwords.csv"
    report_path = tmp_path / "report.json"
    input_path.write_text("Password,Times\nqwerty,0.0\nabcde,1.0\n", encoding="utf-8")

    validate_data_file(str(input_path), str(report_path))

    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "is_valid" in report
    assert "errors" in report
    assert "n_rows" in report
    assert "columns" in report
    assert report["metrics"] == {
        "n_rows": 2,
        "columns": ["Password", "Times"],
        "password_min_length": 5,
        "password_max_length": 6,
        "password_mean_length": 5.5,
        "times_min": 0.0,
        "times_max": 1.0,
        "times_unique_values": [0.0, 1.0],
        "positive_target_share": 0.5,
        "n_duplicate_rows": 0,
    }


def test_validate_password_dataframe_returns_cleaned_required_columns():
    df = pd.DataFrame(
        {
            "Password": ["  qwerty  ", "abcde"],
            "Times": ["0", 1.0],
        }
    )

    is_valid, errors, cleaned_df, metrics = validate_password_dataframe(df)

    assert is_valid is True
    assert errors == []
    assert cleaned_df is not None
    assert list(cleaned_df.columns) == ["Password", "Times"]
    assert cleaned_df["Password"].tolist() == ["qwerty", "abcde"]
    assert cleaned_df["Times"].tolist() == [0.0, 1.0]


def test_validate_password_dataframe_rejects_extra_or_reordered_columns():
    extra_df = pd.DataFrame(
        {"Password": ["qwerty"], "Times": [1.0], "Extra": ["extra"]}
    )
    reordered_df = pd.DataFrame({"Times": [1.0], "Password": ["qwerty"]})

    extra_is_valid, extra_errors, extra_cleaned_df, extra_metrics = (
        validate_password_dataframe(extra_df)
    )
    reordered_is_valid, reordered_errors, reordered_cleaned_df, reordered_metrics = (
        validate_password_dataframe(reordered_df)
    )

    assert extra_is_valid is False
    assert extra_cleaned_df is None
    assert extra_metrics is None
    assert any("exactly match" in error for error in extra_errors)
    assert reordered_is_valid is False
    assert reordered_cleaned_df is None
    assert reordered_metrics is None
    assert any("exactly match" in error for error in reordered_errors)


def test_validate_password_dataframe_rejects_structural_problems():
    df = pd.DataFrame(
        [["password", 1, None], [None, None, None]],
        columns=["Password", "Password", "Empty"],
    )

    is_valid, errors, cleaned_df, metrics = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert any("fully empty rows" in error for error in errors)
    assert any("fully empty columns" in error for error in errors)
    assert any("duplicate columns" in error for error in errors)
    assert any("Times" in error for error in errors)


def test_validate_password_dataframe_rejects_invalid_values():
    df = pd.DataFrame(
        {
            "Password": ["valid", "   ", None, "infinite", "negative"],
            "Times": [1, 2, 3, np.inf, -1],
        }
    )

    is_valid, errors, cleaned_df, metrics = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert any("Password contains missing" in error for error in errors)
    assert any("empty values" in error for error in errors)
    assert any("infinite" in error for error in errors)
    assert any("{0.0, 1.0}" in error for error in errors)


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


def test_large_balanced_target_distribution_is_valid():
    df = pd.DataFrame(
        {
            "Password": make_lowercase_passwords(100),
            "Times": [0.0] * 50 + [1.0] * 50,
        }
    )

    is_valid, errors, cleaned_df, metrics = validate_password_dataframe(df)

    assert is_valid is True
    assert errors == []
    assert cleaned_df is not None
    assert metrics is not None
    assert metrics["positive_target_share"] == 0.5


def test_large_imbalanced_target_distribution_is_invalid():
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


def test_password_with_digits_is_invalid():
    df = pd.DataFrame({"Password": ["password123"], "Times": [1.0]})

    is_valid, errors, cleaned_df, metrics = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert metrics is None
    assert any("lowercase letters a-z" in error for error in errors)
