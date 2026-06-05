import json

from training.validation_thresholds import (
    DEFAULT_MODEL_QUALITY_THRESHOLDS,
    DEFAULT_SCHEMA_THRESHOLDS,
    load_validation_thresholds,
)


def test_load_validation_thresholds_reads_reference_values(tmp_path):
    threshold_path = tmp_path / "validation_thresholds.json"
    threshold_path.write_text(
        json.dumps(
            {
                "schema": {
                    "min_positive_target_share": 0.25,
                    "max_positive_target_share": 0.75,
                    "max_password_length": 64,
                },
                "model_quality": {
                    "max_rmse": 0.2,
                    "max_mae": 0.15,
                    "max_abs_mean_error": 0.08,
                    "max_prediction_target_mean_gap": 0.09,
                },
            }
        ),
        encoding="utf-8",
    )

    thresholds = load_validation_thresholds(threshold_path)

    assert thresholds["schema"] == {
        "min_positive_target_share": 0.25,
        "max_positive_target_share": 0.75,
        "max_password_length": 64,
    }
    assert thresholds["model_quality"] == {
        "max_rmse": 0.2,
        "max_mae": 0.15,
        "max_abs_mean_error": 0.08,
        "max_prediction_target_mean_gap": 0.09,
    }


def test_load_validation_thresholds_uses_defaults_for_missing_file():
    thresholds = load_validation_thresholds("missing-validation-thresholds.json")

    assert thresholds["schema"] == DEFAULT_SCHEMA_THRESHOLDS
    assert thresholds["model_quality"] == DEFAULT_MODEL_QUALITY_THRESHOLDS


def test_load_validation_thresholds_uses_defaults_for_incomplete_file(tmp_path):
    threshold_path = tmp_path / "validation_thresholds.json"
    threshold_path.write_text(
        json.dumps(
            {
                "schema": {"max_password_length": 64},
                "model_quality": {"max_rmse": 0.2},
            }
        ),
        encoding="utf-8",
    )

    thresholds = load_validation_thresholds(threshold_path)

    assert thresholds["schema"] == {
        **DEFAULT_SCHEMA_THRESHOLDS,
        "max_password_length": 64,
    }
    assert thresholds["model_quality"] == {
        **DEFAULT_MODEL_QUALITY_THRESHOLDS,
        "max_rmse": 0.2,
    }
