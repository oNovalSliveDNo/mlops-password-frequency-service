import logging
import os
import re
from typing import Any
from urllib.parse import quote

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response, status
from pydantic import BaseModel, field_validator

app = FastAPI(title="Password Frequency Trigger Service")
logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 5
_SENSITIVE_LOG_KEY_PARTS = ("secret", "credential", "password", "token", "key")
_SENSITIVE_ENV_NAMES = ("GITLAB_TRIGGER_TOKEN", "GITLAB_TOKEN", "AMVERA_PASSWORD")
_REDACTED_LOG_VALUE = "[REDACTED]"
_URL_CREDENTIALS_PATTERN = re.compile(r"(://)([^/\s:@]+):([^@/\s]+)@")


class TriggerRequest(BaseModel):
    data_url: str

    @field_validator("data_url")
    @classmethod
    def validate_data_url(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("data_url must not be empty")

        if not value.startswith(("http://", "https://")):
            raise ValueError("data_url must start with http:// or https://")

        return value


class TriggerResponse(BaseModel):
    status: str
    pipeline_id: int | None = None
    message: str | None = None


class TriggerHealthResponse(BaseModel):
    status: str


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required")
    return value


def _trigger_pipeline_endpoint(gitlab_url: str, project_id: str) -> str:
    encoded_project_id = quote(project_id, safe="")
    return f"{gitlab_url.rstrip('/')}/api/v4/projects/{encoded_project_id}/trigger/pipeline"


def trigger_training_pipeline(data_url: str) -> dict[str, Any]:
    if not data_url:
        raise ValueError("data_url must be provided")

    gitlab_url = _get_required_env("GITLAB_URL")
    project_id = _get_required_env("GITLAB_PROJECT_ID")
    trigger_token = _get_required_env("GITLAB_TRIGGER_TOKEN")
    gitlab_ref = os.getenv("GITLAB_REF", "main")

    response = requests.post(
        _trigger_pipeline_endpoint(gitlab_url, project_id),
        data={
            "token": trigger_token,
            "ref": gitlab_ref,
            "variables[DATA_URL]": data_url,
        },
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    pipeline = response.json()

    result: dict[str, Any] = {"pipeline_id": pipeline["id"]}
    web_url = pipeline.get("web_url")
    if web_url:
        result["web_url"] = web_url

    return result


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


def _run_trigger_training_pipeline(data_url: str) -> None:
    try:
        trigger_training_pipeline(data_url=data_url)
    except Exception as exc:
        _log_endpoint_error("/trigger background task", exc)


@app.get("/health", response_model=TriggerHealthResponse)
def health() -> TriggerHealthResponse:
    return TriggerHealthResponse(status="ok")


@app.post("/trigger", response_model=TriggerResponse)
def trigger(
    request: TriggerRequest,
    background_tasks: BackgroundTasks,
    response: Response,
) -> TriggerResponse:
    try:
        background_tasks.add_task(
            _run_trigger_training_pipeline,
            data_url=request.data_url,
        )
    except Exception as exc:
        _log_endpoint_error("/trigger", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to queue training trigger",
        ) from exc

    response.status_code = status.HTTP_202_ACCEPTED
    return TriggerResponse(status="accepted")
