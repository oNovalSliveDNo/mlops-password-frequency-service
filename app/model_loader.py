import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
import mlflow
from mlflow.tracking import MlflowClient

_model = None
_model_metadata: "ModelLoadMetadata | None" = None
_model_lock = threading.Lock()


@dataclass(frozen=True)
class ModelLoadMetadata:
    model_name: str
    model_alias: str
    requested_model_version: str | None
    loaded_model_version: str | None
    model_uri: str
    reloaded_at: str


def _get_model_name() -> str:
    model_name = os.getenv("MODEL_NAME")

    if not model_name:
        raise RuntimeError(
            "MODEL_NAME environment variable is required to load the MLflow model."
        )

    return model_name


def _get_model_alias() -> str:
    return os.getenv("MODEL_ALIAS", "prod")


def _configure_mlflow_tracking_uri() -> None:
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)


def get_model_uri(expected_model_version: str | None = None) -> str:
    model_name = _get_model_name()
    model_alias = _get_model_alias()

    if expected_model_version:
        return f"models:/{model_name}/{expected_model_version}"

    return f"models:/{model_name}@{model_alias}"


def _read_model_alias_version(model_name: str, model_alias: str) -> str | None:
    model_version = MlflowClient().get_model_version_by_alias(model_name, model_alias)
    version = getattr(model_version, "version", None)
    if version is None:
        return None

    return str(version)


def load_model_from_mlflow(
    expected_model_version: str | None = None,
) -> tuple[object, ModelLoadMetadata]:
    _configure_mlflow_tracking_uri()

    model_name = _get_model_name()
    model_alias = _get_model_alias()
    requested_model_version = (
        str(expected_model_version) if expected_model_version else None
    )
    model_uri = get_model_uri(requested_model_version)

    model = mlflow.pyfunc.load_model(model_uri)
    loaded_model_version = requested_model_version
    if loaded_model_version is None:
        loaded_model_version = _read_model_alias_version(model_name, model_alias)

    metadata = ModelLoadMetadata(
        model_name=model_name,
        model_alias=model_alias,
        requested_model_version=requested_model_version,
        loaded_model_version=loaded_model_version,
        model_uri=model_uri,
        reloaded_at=datetime.now(UTC).isoformat(),
    )

    return model, metadata


def get_model():
    global _model, _model_metadata

    with _model_lock:
        if _model is not None:
            return _model

        _model, _model_metadata = load_model_from_mlflow()
        return _model


def reload_model(expected_model_version: str | None = None) -> ModelLoadMetadata:
    global _model, _model_metadata

    with _model_lock:
        _model, _model_metadata = load_model_from_mlflow(expected_model_version)
        return _model_metadata


def get_model_metadata() -> ModelLoadMetadata | None:
    return _model_metadata


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
    global _model, _model_metadata

    with _model_lock:
        _model = model
        _model_metadata = None
