import pytest
import requests

from app import gitlab_trigger


class FakeResponse:
    def __init__(self, payload, status_error=None):
        self._payload = payload
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error is not None:
            raise self._status_error

    def json(self):
        return self._payload


def _set_required_gitlab_env(monkeypatch):
    monkeypatch.setenv("GITLAB_URL", "https://gitlab.example/")
    monkeypatch.setenv("GITLAB_PROJECT_ID", "group/project")
    monkeypatch.setenv("GITLAB_TRIGGER_TOKEN", "trigger-token")
    monkeypatch.delenv("GITLAB_TOKEN", raising=False)


def test_trigger_training_pipeline_uses_trigger_endpoint_without_project_token(
    monkeypatch,
):
    _set_required_gitlab_env(monkeypatch)
    monkeypatch.setenv("GITLAB_REF", "feature/ref")
    calls = []

    def fake_post(url, data, timeout):
        calls.append({"url": url, "data": data, "timeout": timeout})
        return FakeResponse(
            {
                "id": 123,
                "web_url": "https://gitlab.example/group/project/-/pipelines/123",
            }
        )

    monkeypatch.setattr(gitlab_trigger.requests, "post", fake_post)

    result = gitlab_trigger.trigger_training_pipeline("https://example.com/data.csv")

    assert result == {
        "pipeline_id": 123,
        "web_url": "https://gitlab.example/group/project/-/pipelines/123",
    }
    assert calls == [
        {
            "url": "https://gitlab.example/api/v4/projects/group%2Fproject/trigger/pipeline",
            "data": {
                "token": "trigger-token",
                "ref": "feature/ref",
                "variables[DATA_URL]": "https://example.com/data.csv",
            },
            "timeout": 30,
        }
    ]


def test_trigger_training_pipeline_defaults_to_main_ref(monkeypatch):
    _set_required_gitlab_env(monkeypatch)
    monkeypatch.delenv("GITLAB_REF", raising=False)

    def fake_post(url, data, timeout):
        assert data["ref"] == "main"
        return FakeResponse({"id": 124})

    monkeypatch.setattr(gitlab_trigger.requests, "post", fake_post)

    assert gitlab_trigger.trigger_training_pipeline("https://example.com/data.csv") == {
        "pipeline_id": 124
    }


@pytest.mark.parametrize(
    "missing_env",
    ["GITLAB_URL", "GITLAB_PROJECT_ID", "GITLAB_TRIGGER_TOKEN"],
)
def test_trigger_training_pipeline_requires_minimal_trigger_env(
    monkeypatch, missing_env
):
    _set_required_gitlab_env(monkeypatch)
    monkeypatch.delenv(missing_env, raising=False)

    with pytest.raises(
        RuntimeError, match=f"Environment variable {missing_env} is required"
    ):
        gitlab_trigger.trigger_training_pipeline("https://example.com/data.csv")


def test_trigger_training_pipeline_rejects_missing_data_url(monkeypatch):
    _set_required_gitlab_env(monkeypatch)

    with pytest.raises(ValueError, match="data_url must be provided"):
        gitlab_trigger.trigger_training_pipeline("")


def test_trigger_training_pipeline_propagates_gitlab_http_errors(monkeypatch):
    _set_required_gitlab_env(monkeypatch)
    error = requests.HTTPError("403 Client Error")

    def fake_post(url, data, timeout):
        return FakeResponse({}, status_error=error)

    monkeypatch.setattr(gitlab_trigger.requests, "post", fake_post)

    with pytest.raises(requests.HTTPError, match="403 Client Error"):
        gitlab_trigger.trigger_training_pipeline("https://example.com/data.csv")
