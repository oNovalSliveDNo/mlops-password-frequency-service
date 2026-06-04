import os
from typing import Any

import gitlab


_REQUIRED_ENV_VARS = (
    "GITLAB_URL",
    "GITLAB_PROJECT_ID",
    "GITLAB_TOKEN",
    "GITLAB_TRIGGER_TOKEN",
)


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required")
    return value


def trigger_training_pipeline(data_url: str) -> dict[str, Any]:
    if not data_url:
        raise ValueError("data_url must be provided")

    gitlab_url = _get_required_env("GITLAB_URL")
    project_id = _get_required_env("GITLAB_PROJECT_ID")
    gitlab_token = _get_required_env("GITLAB_TOKEN")
    trigger_token = _get_required_env("GITLAB_TRIGGER_TOKEN")
    gitlab_ref = os.getenv("GITLAB_REF", "main")

    client = gitlab.Gitlab(gitlab_url, private_token=gitlab_token)
    project = client.projects.get(project_id)

    pipeline = project.trigger_pipeline(
        gitlab_ref,
        trigger_token,
        variables={"DATA_URL": data_url},
    )

    result: dict[str, Any] = {"pipeline_id": pipeline.id}
    web_url = getattr(pipeline, "web_url", None)
    if web_url:
        result["web_url"] = web_url

    return result
