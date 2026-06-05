import pytest


class FakeRun:
    def __init__(self):
        self.info = type("RunInfo", (), {"run_id": "run-123"})()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeMlflowClient:
    aliases: list[tuple[str, str, str]] = []
    alias_versions: dict[tuple[str, str], str] = {}

    def search_model_versions(self, query):
        return []

    def set_registered_model_alias(self, model_name, alias, version):
        self.aliases.append((model_name, alias, version))
        self.alias_versions[(model_name, alias)] = str(version)

    def get_model_version_by_alias(self, model_name, alias):
        version = self.alias_versions.get((model_name, alias))
        return type("ModelVersion", (), {"version": version})()


def test_register_model_logs_reports_tests_json_artifact(tmp_path, monkeypatch):
    import importlib
    import sys
    import types

    fake_sklearn = types.ModuleType("mlflow.sklearn")
    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow.sklearn = fake_sklearn
    fake_tracking = types.ModuleType("mlflow.tracking")
    fake_tracking.MlflowClient = FakeMlflowClient

    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.sklearn", fake_sklearn)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", fake_tracking)
    monkeypatch.delitem(sys.modules, "training.register_model", raising=False)

    register_model = importlib.import_module("training.register_model")

    project_root = tmp_path
    reports_dir = project_root / "reports"
    reports_dir.mkdir()
    evidently_report_path = reports_dir / "tests.json"
    evidently_report_path.write_text('{"status": "ok"}', encoding="utf-8")

    logged_artifacts = []

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
    monkeypatch.setenv("MODEL_NAME", "password-frequency-model")
    monkeypatch.setattr(register_model, "_PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        register_model.mlflow, "set_tracking_uri", lambda uri: None, raising=False
    )
    monkeypatch.setattr(
        register_model.mlflow, "set_experiment", lambda name: None, raising=False
    )
    monkeypatch.setattr(
        register_model.mlflow, "start_run", lambda: FakeRun(), raising=False
    )
    monkeypatch.setattr(
        register_model.mlflow, "log_params", lambda params: None, raising=False
    )
    monkeypatch.setattr(
        register_model.mlflow, "log_metrics", lambda metrics: None, raising=False
    )
    monkeypatch.setattr(
        register_model.mlflow, "log_param", lambda key, value: None, raising=False
    )
    monkeypatch.setattr(
        register_model.mlflow, "log_artifact", logged_artifacts.append, raising=False
    )
    monkeypatch.setattr(
        register_model.mlflow.sklearn,
        "log_model",
        lambda **kwargs: type(
            "LogModelResult", (), {"registered_model_version": "7"}
        )(),
        raising=False,
    )
    FakeMlflowClient.aliases = []
    FakeMlflowClient.alias_versions = {}
    monkeypatch.setattr(register_model, "MlflowClient", FakeMlflowClient)

    result = register_model.register_model_in_mlflow(
        model=object(),
        metrics={"rmse_train": 0.1, "n_rows": 4},
    )

    assert str(evidently_report_path) in logged_artifacts
    assert str(project_root / "tests.json") not in logged_artifacts
    assert result == {
        "model_name": "password-frequency-model",
        "model_alias": "prod",
        "model_version": "7",
        "alias_verified": True,
        "verified_model_version": "7",
        "run_id": "run-123",
    }


def test_register_model_fails_when_alias_readback_points_to_old_version(
    tmp_path, monkeypatch
):
    import importlib
    import sys
    import types

    class StaleAliasMlflowClient(FakeMlflowClient):
        def set_registered_model_alias(self, model_name, alias, version):
            self.aliases.append((model_name, alias, version))
            self.alias_versions[(model_name, alias)] = "6"

    fake_sklearn = types.ModuleType("mlflow.sklearn")
    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow.sklearn = fake_sklearn
    fake_tracking = types.ModuleType("mlflow.tracking")
    fake_tracking.MlflowClient = StaleAliasMlflowClient

    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.sklearn", fake_sklearn)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", fake_tracking)
    monkeypatch.delitem(sys.modules, "training.register_model", raising=False)

    register_model = importlib.import_module("training.register_model")

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
    monkeypatch.setenv("MODEL_NAME", "password-frequency-model")
    monkeypatch.setattr(register_model, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(register_model, "_ALIAS_VERIFICATION_ATTEMPTS", 1)
    monkeypatch.setattr(
        register_model.mlflow, "set_tracking_uri", lambda uri: None, raising=False
    )
    monkeypatch.setattr(
        register_model.mlflow, "set_experiment", lambda name: None, raising=False
    )
    monkeypatch.setattr(
        register_model.mlflow, "start_run", lambda: FakeRun(), raising=False
    )
    monkeypatch.setattr(
        register_model.mlflow, "log_metrics", lambda metrics: None, raising=False
    )
    monkeypatch.setattr(
        register_model.mlflow, "log_param", lambda key, value: None, raising=False
    )
    monkeypatch.setattr(
        register_model.mlflow.sklearn,
        "log_model",
        lambda **kwargs: type(
            "LogModelResult", (), {"registered_model_version": "7"}
        )(),
        raising=False,
    )
    StaleAliasMlflowClient.aliases = []
    StaleAliasMlflowClient.alias_versions = {}
    monkeypatch.setattr(register_model, "MlflowClient", StaleAliasMlflowClient)

    with pytest.raises(RuntimeError, match="MLflow alias verification failed"):
        register_model.register_model_in_mlflow(
            model=object(),
            metrics={"rmse_train": 0.1, "n_rows": 4},
        )
