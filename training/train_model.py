from pathlib import Path

import joblib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import FeatureUnion, Pipeline

from training.entropy import TextEntropyTransformer


def train_password_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    passwords = df["Password"]
    times = pd.to_numeric(df["Times"])
    target = np.log10(times)

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
            ("model", Ridge()),
        ]
    )

    pipeline.fit(passwords, target)
    predictions = pipeline.predict(passwords)
    rmse_train = mean_squared_error(target, predictions, squared=False)

    return pipeline, {
        "rmse_train": float(rmse_train),
        "n_rows": int(len(df)),
        "model_type": "Ridge",
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
