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
