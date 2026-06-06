import pytest

import app.model_loader as model_loader
from app.model_loader import ModelLoadMetadata


def test_reload_model_expected_version_mismatch_records_failure_and_preserves_model(
    monkeypatch,
):
    previous_model = object()
    previous_metadata = ModelLoadMetadata(
        model_name="passwords",
        model_alias="prod",
        requested_model_version="11",
        loaded_model_version="11",
        model_uri="models:/passwords/11",
        reloaded_at="2026-06-05T00:00:00+00:00",
    )
    mismatched_metadata = ModelLoadMetadata(
        model_name="passwords",
        model_alias="prod",
        requested_model_version="12",
        loaded_model_version="11",
        model_uri="models:/passwords/12",
        reloaded_at="2026-06-06T00:00:00+00:00",
    )

    monkeypatch.setattr(model_loader, "_model", previous_model)
    monkeypatch.setattr(model_loader, "_model_metadata", previous_metadata)
    monkeypatch.setattr(model_loader, "_last_reload_status", "success")
    monkeypatch.setattr(model_loader, "_last_reload_error", None)
    monkeypatch.setattr(
        model_loader,
        "load_model_from_mlflow",
        lambda expected_model_version: (object(), mismatched_metadata),
    )

    with pytest.raises(RuntimeError, match="Loaded model version does not match"):
        model_loader.reload_model(expected_model_version="12")

    assert model_loader.get_model() is previous_model
    assert model_loader.get_model_metadata() == previous_metadata

    state = model_loader.get_model_state()
    assert state.model_loaded is True
    assert state.loaded_version == "11"
    assert state.model_uri == "models:/passwords/11"
    assert state.last_reload_status == "failed"
    assert state.last_reload_error == "model_reload_failed:RuntimeError"


def test_get_model_state_reports_not_loaded_before_lazy_load(monkeypatch):
    monkeypatch.setattr(model_loader, "_model", None)
    monkeypatch.setattr(model_loader, "_model_metadata", None)
    monkeypatch.setattr(model_loader, "_last_reload_status", "not_loaded")
    monkeypatch.setattr(model_loader, "_last_reload_error", None)

    state = model_loader.get_model_state()

    assert state.model_loaded is False
    assert state.model_name is None
    assert state.loaded_version is None
    assert state.last_reload_status == "not_loaded"
    assert state.last_reload_error is None


def test_get_model_lazy_load_success_records_loaded_state(monkeypatch):
    loaded_model = object()
    loaded_metadata = ModelLoadMetadata(
        model_name="passwords",
        model_alias="prod",
        requested_model_version=None,
        loaded_model_version="15",
        model_uri="models:/passwords@prod",
        reloaded_at="2026-06-06T00:00:00+00:00",
    )

    monkeypatch.setattr(model_loader, "_model", None)
    monkeypatch.setattr(model_loader, "_model_metadata", None)
    monkeypatch.setattr(model_loader, "_last_reload_status", "not_loaded")
    monkeypatch.setattr(model_loader, "_last_reload_error", None)
    monkeypatch.setattr(
        model_loader,
        "load_model_from_mlflow",
        lambda expected_model_version=None: (loaded_model, loaded_metadata),
    )

    assert model_loader.get_model() is loaded_model

    state = model_loader.get_model_state()
    assert state.model_loaded is True
    assert state.model_name == "passwords"
    assert state.loaded_version == "15"
    assert state.model_uri == "models:/passwords@prod"
    assert state.last_reload_status == "success"
    assert state.last_reload_error is None


def test_get_model_lazy_load_failure_records_failed_state(monkeypatch):
    monkeypatch.setattr(model_loader, "_model", None)
    monkeypatch.setattr(model_loader, "_model_metadata", None)
    monkeypatch.setattr(model_loader, "_last_reload_status", "not_loaded")
    monkeypatch.setattr(model_loader, "_last_reload_error", None)

    def fake_load_model_from_mlflow(expected_model_version=None):
        raise RuntimeError("MLflow model is unavailable")

    monkeypatch.setattr(
        model_loader,
        "load_model_from_mlflow",
        fake_load_model_from_mlflow,
    )

    with pytest.raises(RuntimeError, match="MLflow model is unavailable"):
        model_loader.get_model()

    state = model_loader.get_model_state()
    assert state.model_loaded is False
    assert state.loaded_version is None
    assert state.last_reload_status == "failed"
    assert state.last_reload_error == "model_reload_failed:RuntimeError"
