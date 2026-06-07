import pandas as pd

from training.train_model import save_model_artifact, train_password_model


def _training_dataframe():
    return pd.DataFrame(
        {
            "Password": ["qwerty", "123456", "password", "abc123"],
            "Times": [0.1, 0.2, 0.3, 0.4],
        }
    )


def test_train_password_model_returns_model_and_metrics():
    df = _training_dataframe()

    model, metrics = train_password_model(df)

    assert hasattr(model, "predict")
    assert "rmse_train" in metrics

    regressor = model.named_steps["model"]
    assert regressor.n_estimators == 50
    assert regressor.max_depth == 12
    assert regressor.random_state == 42
    assert regressor.n_jobs == 1
    assert metrics["model_n_estimators"] == "50"
    assert metrics["model_max_depth"] == "12"
    assert metrics["model_random_state"] == "42"
    assert metrics["model_n_jobs"] == "1"
    predictions = model.predict(df["Password"])

    assert len(predictions) == len(df)


def test_save_model_artifact(tmp_path):
    df = _training_dataframe()
    model, _ = train_password_model(df)
    output_path = tmp_path / "model.joblib"

    saved_path = save_model_artifact(model, str(output_path))

    assert output_path.exists()
    assert saved_path == str(output_path)
