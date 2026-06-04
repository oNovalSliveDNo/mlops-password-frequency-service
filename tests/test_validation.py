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
