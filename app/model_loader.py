import os
import socket
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
import mlflow
from mlflow.tracking import MlflowClient

_model = None
_model_metadata: "ModelLoadMetadata | None" = None
_last_reload_status = "not_loaded"
_last_reload_error: str | None = None
_model_lock = threading.Lock()


def get_instance_id() -> str:
    """Return a stable diagnostic identifier for this service process."""
    return (
        os.getenv("INSTANCE_ID")
        or os.getenv("SERVICE_INSTANCE_ID")
        or socket.gethostname()
    )


@dataclass(frozen=True)
class ModelLoadMetadata:
    model_name: str
    model_alias: str
    requested_model_version: str | None
    loaded_model_version: str | None
    model_uri: str
    reloaded_at: str
    instance_id: str = ""


@dataclass(frozen=True)
class ModelServiceState:
    model_loaded: bool
    model_name: str | None
    model_alias: str | None
    loaded_version: str | None
    model_uri: str | None
    loaded_at: str | None
    last_reload_status: str
    last_reload_error: str | None
    instance_id: str = ""


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
        instance_id=get_instance_id(),
    )

    return model, metadata


def _safe_reload_error(exc: Exception) -> str:
    """Return a short public-safe category for the last reload failure."""
    return f"model_reload_failed:{type(exc).__name__}"


def get_model():
    global _last_reload_error, _last_reload_status, _model, _model_metadata

    with _model_lock:
        if _model is not None:
            return _model

        try:
            _model, _model_metadata = load_model_from_mlflow()
        except Exception as exc:
            _last_reload_status = "failed"
            _last_reload_error = _safe_reload_error(exc)
            raise

        _last_reload_status = "success"
        _last_reload_error = None
        return _model


def _ensure_expected_version_loaded(metadata: ModelLoadMetadata) -> None:
    if (
        metadata.requested_model_version is not None
        and metadata.loaded_model_version != metadata.requested_model_version
    ):
        raise RuntimeError(
            "Loaded model version does not match requested version: "
            f"{metadata.loaded_model_version!r} != {metadata.requested_model_version!r}"
        )


def reload_model(expected_model_version: str | None = None) -> ModelLoadMetadata:
    global _last_reload_error, _last_reload_status, _model, _model_metadata

    try:
        loaded_model, loaded_metadata = load_model_from_mlflow(expected_model_version)
        _ensure_expected_version_loaded(loaded_metadata)
    except Exception as exc:
        with _model_lock:
            _last_reload_status = "failed"
            _last_reload_error = _safe_reload_error(exc)
        raise

    with _model_lock:
        _model = loaded_model
        _model_metadata = loaded_metadata
        _last_reload_status = "success"
        _last_reload_error = None
        return _model_metadata


def get_model_metadata() -> ModelLoadMetadata | None:
    with _model_lock:
        return _model_metadata


def get_model_state() -> ModelServiceState:
    with _model_lock:
        return ModelServiceState(
            instance_id=get_instance_id(),
            model_loaded=_model is not None,
            model_name=_model_metadata.model_name
            if _model_metadata is not None
            else None,
            model_alias=_model_metadata.model_alias
            if _model_metadata is not None
            else None,
            loaded_version=(
                _model_metadata.loaded_model_version
                if _model_metadata is not None
                else None
            ),
            model_uri=_model_metadata.model_uri
            if _model_metadata is not None
            else None,
            loaded_at=_model_metadata.reloaded_at
            if _model_metadata is not None
            else None,
            last_reload_status=_last_reload_status,
            last_reload_error=_last_reload_error,
        )


def is_model_loaded() -> bool:
    with _model_lock:
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
    global _last_reload_error, _last_reload_status, _model, _model_metadata

    with _model_lock:
        _model = model
        _model_metadata = None
        _last_reload_status = "success" if model is not None else "not_loaded"
        _last_reload_error = None
