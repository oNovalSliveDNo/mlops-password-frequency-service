import numpy as np
import pandas as pd

from training.model_quality_validation import validate_model_quality_with_prod_model
from training.validation_thresholds import DEFAULT_MODEL_QUALITY_THRESHOLDS


class DummyModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, passwords):
        return self.predictions[: len(passwords)]


def _quality_df():
    return pd.DataFrame(
        {
            "Password": ["a", "b", "c", "d"],
            "Times": [0.0, 1.0, 0.0, 1.0],
        }
    )


def test_model_quality_accepts_good_predictions():
    df = _quality_df()
    predictions = np.log10(df["Times"].to_numpy() + 1)

    result = validate_model_quality_with_prod_model(
        df,
        prod_model=DummyModel(predictions),
    )

    assert result.is_valid
    assert result.errors == []
    assert np.isclose(result.metrics["rmse"], 0)
    assert {"target_log", "prediction", "prediction_error"}.issubset(
        result.scored_df.columns
    )


def test_model_quality_rejects_bad_predictions():
    df = _quality_df()
    target_log = np.log10(df["Times"].to_numpy() + 1)
    predictions = target_log[::-1]

    result = validate_model_quality_with_prod_model(
        df,
        prod_model=DummyModel(predictions),
    )

    assert not result.is_valid
    assert result.errors
    assert result.metrics["rmse"] > DEFAULT_MODEL_QUALITY_THRESHOLDS["max_rmse"]
    assert result.metrics["mae"] > DEFAULT_MODEL_QUALITY_THRESHOLDS["max_mae"]


def test_model_quality_uses_log_target_not_raw_times():
    df = pd.DataFrame(
        {
            "Password": ["a"],
            "Times": [1.0],
        }
    )
    prediction = [np.log10(2)]

    result = validate_model_quality_with_prod_model(
        df,
        prod_model=DummyModel(prediction),
    )

    assert result.is_valid or np.isclose(result.metrics["rmse"], 0)
