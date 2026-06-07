from pathlib import Path

import joblib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import FeatureUnion, Pipeline

from training.entropy import TextEntropyTransformer


RANDOM_FOREST_PARAMS = {
    "n_estimators": 50,
    "max_depth": 12,
    "random_state": 42,
    "n_jobs": 1,
}


def train_password_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    passwords = df["Password"]
    times = pd.to_numeric(df["Times"])
    target = np.log10(times + 1)

    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "text",
                            Pipeline(
                                [
                                    (
                                        "count",
                                        CountVectorizer(
                                            analyzer="char", ngram_range=(1, 3)
                                        ),
                                    ),
                                    ("tfidf", TfidfTransformer()),
                                ]
                            ),
                        ),
                        ("entropy", TextEntropyTransformer()),
                    ]
                ),
            ),
            ("model", RandomForestRegressor(**RANDOM_FOREST_PARAMS)),
        ]
    )

    pipeline.fit(passwords, target)
    predictions = pipeline.predict(passwords)
    rmse_train = root_mean_squared_error(target, predictions)
    mae_train = mean_absolute_error(target, predictions)
    errors = predictions - target
    abs_mean_error_train = abs(float(np.mean(errors)))
    prediction_target_mean_gap_train = abs(
        float(np.mean(predictions) - np.mean(target))
    )

    return pipeline, {
        "rmse_train": float(rmse_train),
        "mae_train": float(mae_train),
        "abs_mean_error_train": abs_mean_error_train,
        "prediction_target_mean_gap_train": prediction_target_mean_gap_train,
        "n_rows": int(len(df)),
        "model_type": "RandomForestRegressor",
        "model_n_estimators": str(RANDOM_FOREST_PARAMS["n_estimators"]),
        "model_max_depth": str(RANDOM_FOREST_PARAMS["max_depth"]),
        "model_random_state": str(RANDOM_FOREST_PARAMS["random_state"]),
        "model_n_jobs": str(RANDOM_FOREST_PARAMS["n_jobs"]),
        "ngram_min": 1,
        "ngram_max": 3,
        "tfidf": True,
    }


def save_model_artifact(model, output_path: str = "artifacts/model.joblib") -> str:
    """Save a local debug/CI model artifact without replacing MLflow registry use.

    Production model registration remains in MLflow, where the ``prod`` alias
    identifies the production model. The local artifact is intended only for
    debugging and CI artifact collection.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return str(path)
