import os

from fastapi import FastAPI, Header, HTTPException, Response, status

from app.gitlab_trigger import trigger_training_pipeline
from app.model_loader import get_model_state, predict_passwords, reload_model
from app.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReloadRequest,
    ReloadResponse,
    TriggerRequest,
    TriggerResponse,
)

app = FastAPI(title="Password Frequency Service")


def _current_health_response() -> HealthResponse:
    model_state = get_model_state()
    return HealthResponse(
        status="ok",
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


@app.get("/model_state", response_model=HealthResponse)
def model_state() -> HealthResponse:
    """Expose the loaded model metadata for post-reload serving checks."""
    return _current_health_response()


@app.get("/model_status", response_model=HealthResponse)
def model_status() -> HealthResponse:
    """Expose the loaded model metadata using a status-oriented alias."""
    return _current_health_response()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        predictions = predict_passwords(request.Password)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is unavailable: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to generate predictions: {exc}",
        ) from exc

    return PredictResponse(Times=predictions)


@app.post("/trigger", response_model=TriggerResponse)
def trigger(request: TriggerRequest, response: Response) -> TriggerResponse:
    try:
        result = trigger_training_pipeline(data_url=request.data_url)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid training trigger request: {exc}",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GitLab trigger is not configured correctly: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to start GitLab training pipeline: {exc}",
        ) from exc

    response.status_code = status.HTTP_202_ACCEPTED
    pipeline_id = result.get("pipeline_id")
    message = "Training pipeline started"
    web_url = result.get("web_url")
    if web_url:
        message = f"Training pipeline started: {web_url}"

    return TriggerResponse(
        status="started",
        pipeline_id=pipeline_id,
        message=message,
    )


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
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model reload failed: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unexpected model reload error: {exc}",
        ) from exc

    return ReloadResponse(
        status="model_reloaded",
        model_name=load_metadata.model_name,
        model_alias=load_metadata.model_alias,
        requested_model_version=load_metadata.requested_model_version,
        loaded_model_version=load_metadata.loaded_model_version,
        model_uri=load_metadata.model_uri,
        reloaded_at=load_metadata.reloaded_at,
    )
