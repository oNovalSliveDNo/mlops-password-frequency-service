import os
from typing import Any
from urllib.parse import quote

import requests


_REQUIRED_ENV_VARS = (
    "GITLAB_URL",
    "GITLAB_PROJECT_ID",
    "GITLAB_TRIGGER_TOKEN",
)
_REQUEST_TIMEOUT_SECONDS = 5


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
