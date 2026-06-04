import os

from fastapi import FastAPI, Header, HTTPException, Response, status

from app.gitlab_trigger import trigger_training_pipeline
from app.model_loader import is_model_loaded, predict_passwords, reload_model
from app.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReloadResponse,
    TriggerRequest,
    TriggerResponse,
)

app = FastAPI(title="Password Frequency Service")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=is_model_loaded())


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

    try:
        reload_model()
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
        model_name=os.getenv("MODEL_NAME"),
        model_alias=os.getenv("MODEL_ALIAS", "prod"),
    )
