import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

_model = None
_model_metadata: "ModelLoadMetadata | None" = None
_last_reload_status = "not_loaded"
_last_reload_error: str | None = None
_model_lock = threading.Lock()
_auto_reload_lock = threading.Lock()
_alias_version_cache_lock = threading.Lock()
_cached_alias_version: str | None = None
_alias_version_checked_at = 0.0


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


def get_current_model_alias_version(
    model_name: str | None = None,
    model_alias: str | None = None,
) -> str | None:
    """Read the current MLflow model version behind the configured alias."""
    _configure_mlflow_tracking_uri()
    resolved_model_name = model_name or _get_model_name()
    resolved_model_alias = model_alias or _get_model_alias()
    model_version = MlflowClient().get_model_version_by_alias(
        resolved_model_name, resolved_model_alias
    )
    version = getattr(model_version, "version", None)
    if version is None:
        return None

    return str(version)


def _get_alias_version_cache_ttl_seconds() -> float:
    value = os.getenv("MODEL_ALIAS_CHECK_TTL_SECONDS", "5")
    try:
        ttl_seconds = float(value)
    except ValueError:
        return 5.0

    return max(0.0, ttl_seconds)


def _read_model_alias_version(model_name: str, model_alias: str) -> str | None:
    return get_current_model_alias_version(model_name, model_alias)


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


def _read_current_alias_version_with_ttl() -> str | None:
    global _alias_version_checked_at, _cached_alias_version

    ttl_seconds = _get_alias_version_cache_ttl_seconds()
    now = time.monotonic()
    with _alias_version_cache_lock:
        if (
            _cached_alias_version is not None
            and ttl_seconds > 0
            and now - _alias_version_checked_at < ttl_seconds
        ):
            return _cached_alias_version

    current_alias_version = get_current_model_alias_version()
    with _alias_version_cache_lock:
        _cached_alias_version = current_alias_version
        _alias_version_checked_at = time.monotonic()

    return current_alias_version


def _snapshot_loaded_model() -> tuple[object | None, ModelLoadMetadata | None]:
    with _model_lock:
        return _model, _model_metadata


def _remember_loaded_alias_version(metadata: ModelLoadMetadata | None) -> None:
    global _alias_version_checked_at, _cached_alias_version

    if metadata is None or metadata.loaded_model_version is None:
        return

    with _alias_version_cache_lock:
        _cached_alias_version = metadata.loaded_model_version
        _alias_version_checked_at = time.monotonic()


def _mark_alias_check_failed(exc: Exception) -> None:
    global _last_reload_error, _last_reload_status

    with _model_lock:
        _last_reload_status = "failed"
        _last_reload_error = _safe_reload_error(exc)


def ensure_current_alias_model_loaded() -> None:
    """Auto-reload the loaded model when the configured alias moves."""
    loaded_model, loaded_metadata = _snapshot_loaded_model()
    if loaded_model is None:
        return
    if loaded_metadata is None or loaded_metadata.loaded_model_version is None:
        return

    if not _auto_reload_lock.acquire(blocking=False):
        return

    try:
        _, loaded_metadata = _snapshot_loaded_model()
        if loaded_metadata is None or loaded_metadata.loaded_model_version is None:
            return

        try:
            current_alias_version = _read_current_alias_version_with_ttl()
        except Exception as exc:
            logger.warning(
                "auto_reload_on_predict_alias_check_failed: old_version=%s, instance_id=%s",
                loaded_metadata.loaded_model_version,
                get_instance_id(),
            )
            _mark_alias_check_failed(exc)
            return

        if current_alias_version is None:
            return
        if current_alias_version == loaded_metadata.loaded_model_version:
            return

        logger.info(
            "auto_reload_on_predict: old_version=%s, new_version=%s, instance_id=%s",
            loaded_metadata.loaded_model_version,
            current_alias_version,
            get_instance_id(),
        )
        try:
            reload_model(expected_model_version=current_alias_version)
        except Exception as exc:
            logger.warning(
                "auto_reload_on_predict_failed: old_version=%s, new_version=%s, instance_id=%s",
                loaded_metadata.loaded_model_version,
                current_alias_version,
                get_instance_id(),
            )
            _mark_alias_check_failed(exc)
    finally:
        _auto_reload_lock.release()


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
        _remember_loaded_alias_version(_model_metadata)
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
        _remember_loaded_alias_version(_model_metadata)
        return _model_metadata


def get_model_metadata() -> ModelLoadMetadata | None:
    with _model_lock:
        return _model_metadata


def get_model_state() -> ModelServiceState:
    with _model_lock:
        return ModelServiceState(
            instance_id=get_instance_id(),
            model_loaded=_model is not None,
            model_name=(
                _model_metadata.model_name if _model_metadata is not None else None
            ),
            model_alias=(
                _model_metadata.model_alias if _model_metadata is not None else None
            ),
            loaded_version=(
                _model_metadata.loaded_model_version
                if _model_metadata is not None
                else None
            ),
            model_uri=(
                _model_metadata.model_uri if _model_metadata is not None else None
            ),
            loaded_at=(
                _model_metadata.reloaded_at if _model_metadata is not None else None
            ),
            last_reload_status=_last_reload_status,
            last_reload_error=_last_reload_error,
        )


def is_model_loaded() -> bool:
    with _model_lock:
        return _model is not None


def predict_passwords(passwords: list[str]) -> list[float]:
    ensure_current_alias_model_loaded()
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
    global _alias_version_checked_at, _cached_alias_version
    global _last_reload_error, _last_reload_status, _model, _model_metadata

    with _alias_version_cache_lock:
        _cached_alias_version = None
        _alias_version_checked_at = 0.0

    with _model_lock:
        _model = model
        _model_metadata = None
        _last_reload_status = "success" if model is not None else "not_loaded"
        _last_reload_error = None
