import importlib.util
import sys
import types
from typing import Optional, Any, Callable


class RequestException(Exception):
    pass


class HTTPError(RequestException):
    def __init__(self, *args, response=None, **kwargs):
        super().__init__(*args)
        self.response = response


class ConnectionError(RequestException):
    pass


class FakeRequestsModule(types.ModuleType):
    """Типизированная заглушка для модуля requests."""

    RequestException: type[RequestException]
    HTTPError: type[HTTPError]
    ConnectionError: type[ConnectionError]
    post: Optional[Callable[..., Any]]

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.RequestException = RequestException
        self.HTTPError = HTTPError
        self.ConnectionError = ConnectionError
        self.post = None


if importlib.util.find_spec("requests") is None:
    sys.modules["requests"] = FakeRequestsModule("requests")


class FakePyfuncModule(types.ModuleType):
    def load_model(self, model_uri: str) -> Any:
        raise RuntimeError(f"mlflow is not installed; cannot load {model_uri}")


class FakeMlflowModule(types.ModuleType):
    pyfunc: FakePyfuncModule

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.pyfunc = FakePyfuncModule("mlflow.pyfunc")

    def set_tracking_uri(self, mlflow_tracking_uri: str) -> None:
        return None


class FakeMlflowClient:
    def get_model_version_by_alias(self, model_name: str, model_alias: str) -> Any:
        raise RuntimeError(
            "mlflow is not installed; cannot read model version alias "
            f"{model_name}@{model_alias}"
        )


class FakeMlflowTrackingModule(types.ModuleType):
    MlflowClient: type[FakeMlflowClient]

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.MlflowClient = FakeMlflowClient


if importlib.util.find_spec("mlflow") is None:
    fake_mlflow = FakeMlflowModule("mlflow")
    sys.modules["mlflow"] = fake_mlflow
    sys.modules["mlflow.pyfunc"] = fake_mlflow.pyfunc
    sys.modules["mlflow.tracking"] = FakeMlflowTrackingModule("mlflow.tracking")
