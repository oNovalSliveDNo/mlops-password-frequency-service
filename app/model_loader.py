import os
import threading

import mlflow


_model = None
_model_lock = threading.Lock()


def get_model_uri() -> str:
    model_name = os.getenv("MODEL_NAME")
    model_alias = os.getenv("MODEL_ALIAS", "prod")

    if not model_name:
        raise RuntimeError(
            "MODEL_NAME environment variable is required to load the MLflow model."
        )

    return f"models:/{model_name}@{model_alias}"


def load_model_from_mlflow():
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    return mlflow.pyfunc.load_model(get_model_uri())


def get_model():
    global _model

    with _model_lock:
        if _model is not None:
            return _model

        _model = load_model_from_mlflow()
        return _model


def reload_model():
    global _model

    with _model_lock:
        _model = load_model_from_mlflow()
        return _model


def is_model_loaded() -> bool:
    return _model is not None


def predict_passwords(passwords: list[str]) -> list[float]:
    model = get_model()
    predictions = model.predict(passwords)

    if hasattr(predictions, "to_numpy"):
        values = predictions.to_numpy().ravel().tolist()
    elif hasattr(predictions, "ravel"):
        values = predictions.ravel().tolist()
    elif hasattr(predictions, "tolist"):
        values = predictions.tolist()
    else:
        values = list(predictions)

    return [float(value) for value in values]


def set_model_for_tests(model):
    global _model

    with _model_lock:
        _model = model
