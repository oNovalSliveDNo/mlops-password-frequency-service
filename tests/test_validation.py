import json


import numpy as np
import pandas as pd

from training.validate_data import validate_password_dataframe


def test_good_dataframe_is_valid():
    df = pd.DataFrame({"Password": ["qwerty", "123456"], "Times": [0.1, 0.2]})

    is_valid, errors, cleaned_df = validate_password_dataframe(df)

    assert is_valid is True
    assert errors == []
    assert cleaned_df is not None


def test_missing_password_column_is_invalid():
    df = pd.DataFrame({"Times": [1.0]})

    is_valid, errors, cleaned_df = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert any("Password" in error for error in errors)


def test_missing_times_column_is_invalid():
    df = pd.DataFrame({"Password": ["qwerty"]})

    is_valid, errors, cleaned_df = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert any("Times" in error for error in errors)


def test_empty_password_is_invalid():
    df = pd.DataFrame({"Password": ["   "], "Times": [1.0]})

    is_valid, errors, cleaned_df = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert any("empty values" in error for error in errors)


def test_non_positive_times_is_invalid():
    df = pd.DataFrame({"Password": ["qwerty", "123456"], "Times": [0, -1]})

    is_valid, errors, cleaned_df = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert any("positive" in error for error in errors)


def test_non_numeric_times_is_invalid():
    df = pd.DataFrame({"Password": ["valid"], "Times": ["not-a-number"]})

    is_valid, errors, cleaned_df = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert any("non-numeric" in error for error in errors)


def test_validate_data_file_writes_report(tmp_path):
    from training.validate_data import validate_data_file

    input_path = tmp_path / "passwords.csv"
    report_path = tmp_path / "report.json"
    input_path.write_text("Password,Times\nqwerty,0.1\n123456,0.2\n", encoding="utf-8")

    validate_data_file(str(input_path), str(report_path))

    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "is_valid" in report
    assert "errors" in report
    assert "n_rows" in report
    assert "columns" in report


def test_validate_password_dataframe_returns_cleaned_required_columns():
    df = pd.DataFrame(
        {
            "Password": ["  qwerty  ", 12345],
            "Times": ["10", 2.5],
            "Extra": ["ignored", "ignored"],
        }
    )

    is_valid, errors, cleaned_df = validate_password_dataframe(df)

    assert is_valid is True
    assert errors == []
    assert cleaned_df is not None
    assert list(cleaned_df.columns) == ["Password", "Times"]
    assert cleaned_df["Password"].tolist() == ["qwerty", "12345"]
    assert cleaned_df["Times"].tolist() == [10.0, 2.5]


def test_validate_password_dataframe_rejects_structural_problems():
    df = pd.DataFrame(
        [["password", 1, None], [None, None, None]],
        columns=["Password", "Password", "Empty"],
    )

    is_valid, errors, cleaned_df = validate_password_dataframe(df)

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

    is_valid, errors, cleaned_df = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert any("Password contains missing" in error for error in errors)
    assert any("empty values" in error for error in errors)
    assert any("infinite" in error for error in errors)
    assert any("positive" in error for error in errors)


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
        '  "columns": []\n'
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
