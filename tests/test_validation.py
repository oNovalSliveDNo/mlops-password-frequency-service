import numpy as np
import pandas as pd

from training.validate_data import validate_password_dataframe


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


def test_validate_password_dataframe_rejects_non_numeric_times():
    df = pd.DataFrame({"Password": ["valid"], "Times": ["not-a-number"]})

    is_valid, errors, cleaned_df = validate_password_dataframe(df)

    assert is_valid is False
    assert cleaned_df is None
    assert any("non-numeric" in error for error in errors)


def test_validate_data_file_writes_valid_report(tmp_path):
    from training.validate_data import ValidationResult, validate_data_file

    input_path = tmp_path / "passwords.csv"
    report_path = tmp_path / "report.json"
    input_path.write_text(
        "Password,Times,Extra\n  abc  ,3,ignored\ndef,4,ignored\n", encoding="utf-8"
    )

    result = validate_data_file(str(input_path), str(report_path))

    assert result == ValidationResult(True, [], 2, ["Password", "Times"])
    assert report_path.read_text(encoding="utf-8") == (
        "{\n"
        '  "is_valid": true,\n'
        '  "errors": [],\n'
        '  "n_rows": 2,\n'
        '  "columns": [\n'
        '    "Password",\n'
        '    "Times"\n'
        "  ]\n"
        "}"
    )


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
