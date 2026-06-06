import logging
import os
import re
from typing import TYPE_CHECKING, Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Response, status

from app.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReadinessResponse,
    ReloadRequest,
    ReloadResponse,
    TriggerRequest,
    TriggerResponse,
)

if TYPE_CHECKING:
    from app.model_loader import PredictDiagnostics

# The service intentionally uses lazy model loading: the MLflow model is
# loaded by get_model() on the first prediction or by an explicit reload request.
# /health is therefore a liveness probe and /ready is the readiness probe.
MODEL_LOADING_MODE = "lazy"

app = FastAPI(title="Password Frequency Serving Service")
logger = logging.getLogger(__name__)

_SENSITIVE_LOG_KEY_PARTS = ("secret", "credential", "password", "token", "key")
_SENSITIVE_ENV_NAMES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "MLFLOW_TRACKING_PASSWORD",
    "DOCKERHUB_TOKEN",
    "SERVICE_RELOAD_SECRET",
    "SERVICE_RELOAD_URL",
    "GITLAB_TRIGGER_TOKEN",
    "AMVERA_PASSWORD",
)
_REDACTED_LOG_VALUE = "[REDACTED]"
_URL_CREDENTIALS_PATTERN = re.compile(r"(://)([^/\s:@]+):([^@/\s]+)@")


def get_model_state():
    """Lazy proxy to avoid importing ML dependencies during FastAPI startup."""
    from app.model_loader import get_model_state as _get_model_state

    return _get_model_state()


def get_predict_diagnostics(password_count: int):
    """Lazy proxy to avoid importing ML dependencies until prediction handling."""
    from app.model_loader import get_predict_diagnostics as _get_predict_diagnostics

    return _get_predict_diagnostics(password_count=password_count)


def predict_passwords(passwords: list[str]) -> list[float]:
    """Lazy proxy to avoid importing ML dependencies until prediction handling."""
    from app.model_loader import predict_passwords as _predict_passwords

    return _predict_passwords(passwords)


def reload_model(expected_model_version: str | None = None):
    """Lazy proxy to avoid importing ML dependencies until model reload handling."""
    from app.model_loader import reload_model as _reload_model

    return _reload_model(expected_model_version=expected_model_version)


def trigger_training_pipeline(data_url: str):
    """Lazy proxy for compatibility with tests and the legacy /trigger endpoint."""
    from app.gitlab_trigger import (
        trigger_training_pipeline as _trigger_training_pipeline,
    )

    return _trigger_training_pipeline(data_url=data_url)


def _is_sensitive_log_key(key: str) -> bool:
    normalized_key = key.lower()
    return any(part in normalized_key for part in _SENSITIVE_LOG_KEY_PARTS)


def _secret_env_values() -> tuple[str, ...]:
    return tuple(
        value for env_name in _SENSITIVE_ENV_NAMES if (value := os.getenv(env_name))
    )


def _sanitize_for_log(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                _REDACTED_LOG_VALUE
                if _is_sensitive_log_key(str(key))
                else _sanitize_for_log(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_for_log(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_for_log(item) for item in value)
    if isinstance(value, str):
        sanitized = value
        for secret_value in _secret_env_values():
            sanitized = sanitized.replace(secret_value, _REDACTED_LOG_VALUE)
        return _URL_CREDENTIALS_PATTERN.sub(
            rf"\1{_REDACTED_LOG_VALUE}:{_REDACTED_LOG_VALUE}@", sanitized
        )
    return value


def _log_endpoint_error(endpoint: str, exc: Exception) -> None:
    logger.error(
        "%s failed with %s: %s",
        endpoint,
        type(exc).__name__,
        _sanitize_for_log(str(exc)),
    )


def _current_health_response() -> HealthResponse:
    model_state = get_model_state()
    return HealthResponse(
        status="ok",
        instance_id=model_state.instance_id,
        model_loaded=model_state.model_loaded,
        model_name=model_state.model_name,
        model_alias=model_state.model_alias,
        loaded_version=model_state.loaded_version,
        model_uri=model_state.model_uri,
        loaded_at=model_state.loaded_at,
        last_reload_status=model_state.last_reload_status,
        last_reload_error=model_state.last_reload_error,
    )


def _current_readiness_response() -> ReadinessResponse:
    model_state = get_model_state()
    return ReadinessResponse(
        status="ready" if model_state.model_loaded else "not_ready",
        instance_id=model_state.instance_id,
        model_loaded=model_state.model_loaded,
        model_name=model_state.model_name,
        model_alias=model_state.model_alias,
        loaded_version=model_state.loaded_version,
        model_uri=model_state.model_uri,
        loaded_at=model_state.loaded_at,
        last_reload_status=model_state.last_reload_status,
        last_reload_error=model_state.last_reload_error,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return _current_health_response()


@app.get("/ready", response_model=ReadinessResponse)
def ready(response: Response) -> ReadinessResponse:
    readiness_response = _current_readiness_response()
    if not readiness_response.model_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return readiness_response


@app.get("/model_state", response_model=HealthResponse)
def model_state() -> HealthResponse:
    """Expose the loaded model metadata for post-reload serving checks."""
    return _current_health_response()


@app.get("/model_status", response_model=HealthResponse)
def model_status() -> HealthResponse:
    """Expose the loaded model metadata using a status-oriented alias."""
    return _current_health_response()


def _set_model_headers(
    response: Response, diagnostics: "PredictDiagnostics | None" = None
) -> None:
    if diagnostics is None:
        model_state = get_model_state()
        response.headers["X-Instance-ID"] = model_state.instance_id
        response.headers["X-Model-Version"] = model_state.loaded_version or ""
        return

    response.headers["X-Instance-ID"] = diagnostics.instance_id
    response.headers["X-Model-Version"] = diagnostics.loaded_version or ""


def _log_predict_diagnostics(diagnostics: "PredictDiagnostics") -> None:
    logger.info(
        (
            "predict_completed: instance_id=%s loaded_version=%s "
            "model_uri=%s password_count=%s"
        ),
        _sanitize_for_log(diagnostics.instance_id),
        _sanitize_for_log(diagnostics.loaded_version),
        _sanitize_for_log(diagnostics.model_uri),
        diagnostics.password_count,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, response: Response) -> PredictResponse:
    password_count = len(request.Password)
    try:
        predictions = predict_passwords(request.Password)
    except RuntimeError as exc:
        _log_endpoint_error("/predict", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently unavailable",
        ) from exc
    except Exception as exc:
        _log_endpoint_error("/predict", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to generate predictions",
        ) from exc

    diagnostics = get_predict_diagnostics(password_count=password_count)
    _log_predict_diagnostics(diagnostics)
    _set_model_headers(response, diagnostics)
    return PredictResponse(Times=predictions)


def _run_trigger_training_pipeline(data_url: str) -> None:
    try:
        trigger_training_pipeline(data_url=data_url)
    except Exception as exc:
        _log_endpoint_error("/trigger background task", exc)


@app.post("/trigger", response_model=TriggerResponse)
def trigger(
    request: TriggerRequest,
    background_tasks: BackgroundTasks,
    response: Response,
) -> TriggerResponse:
    background_tasks.add_task(_run_trigger_training_pipeline, data_url=request.data_url)
    response.status_code = status.HTTP_202_ACCEPTED
    return TriggerResponse(status="accepted")


@app.post("/reload_model", response_model=ReloadResponse)
def reload_model_endpoint(
    request: ReloadRequest | None = None,
    x_service_token: str | None = Header(default=None, alias="X-Service-Token"),
) -> ReloadResponse:
    expected_token = os.getenv("SERVICE_RELOAD_SECRET")
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SERVICE_RELOAD_SECRET environment variable is not configured",
        )

    if x_service_token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid service token",
        )

    model_name = os.getenv("MODEL_NAME")
    model_alias = os.getenv("MODEL_ALIAS", "prod")
    if request is not None:
        if request.model_name is not None and request.model_name != model_name:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Requested model_name does not match service configuration: "
                    f"{request.model_name!r} != {model_name!r}"
                ),
            )
        if request.model_alias is not None and request.model_alias != model_alias:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Requested model_alias does not match service configuration: "
                    f"{request.model_alias!r} != {model_alias!r}"
                ),
            )

    expected_model_version = (
        request.expected_model_version if request is not None else None
    )

    try:
        load_metadata = reload_model(expected_model_version=expected_model_version)
    except RuntimeError as exc:
        _log_endpoint_error("/reload_model", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model reload failed",
        ) from exc
    except Exception as exc:
        _log_endpoint_error("/reload_model", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unexpected model reload error",
        ) from exc

    return ReloadResponse(
        status="model_reloaded",
        instance_id=load_metadata.instance_id,
        model_name=load_metadata.model_name,
        model_alias=load_metadata.model_alias,
        requested_model_version=load_metadata.requested_model_version,
        loaded_model_version=load_metadata.loaded_model_version,
        model_uri=load_metadata.model_uri,
        reloaded_at=load_metadata.reloaded_at,
    )
