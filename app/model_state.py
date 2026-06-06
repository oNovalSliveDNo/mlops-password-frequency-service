import os
import socket
import threading
from dataclasses import dataclass


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


@dataclass(frozen=True)
class PredictDiagnostics:
    instance_id: str
    loaded_version: str | None
    model_uri: str | None
    password_count: int


_state_lock = threading.Lock()
_model_loaded = False
_model_metadata: ModelLoadMetadata | None = None
_last_reload_status = "not_loaded"
_last_reload_error: str | None = None


def get_instance_id() -> str:
    """Return a stable diagnostic identifier for this service process."""
    return (
        os.getenv("INSTANCE_ID")
        or os.getenv("SERVICE_INSTANCE_ID")
        or socket.gethostname()
    )


def set_model_state(
    *,
    model_loaded: bool,
    model_metadata: ModelLoadMetadata | None,
    last_reload_status: str,
    last_reload_error: str | None,
) -> None:
    """Update the public serving state without importing ML dependencies."""
    global _last_reload_error, _last_reload_status, _model_loaded, _model_metadata

    with _state_lock:
        _model_loaded = model_loaded
        _model_metadata = model_metadata
        _last_reload_status = last_reload_status
        _last_reload_error = last_reload_error


def get_model_state() -> ModelServiceState:
    """Return public serving state without importing MLflow or model code."""
    with _state_lock:
        metadata = _model_metadata
        return ModelServiceState(
            instance_id=get_instance_id(),
            model_loaded=_model_loaded,
            model_name=metadata.model_name if metadata is not None else None,
            model_alias=metadata.model_alias if metadata is not None else None,
            loaded_version=(
                metadata.loaded_model_version if metadata is not None else None
            ),
            model_uri=metadata.model_uri if metadata is not None else None,
            loaded_at=metadata.reloaded_at if metadata is not None else None,
            last_reload_status=_last_reload_status,
            last_reload_error=_last_reload_error,
        )
