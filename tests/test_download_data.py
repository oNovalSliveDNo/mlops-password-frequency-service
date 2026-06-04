import pytest
import requests

from training.download_data import download_data


def test_download_data_success(monkeypatch, tmp_path):
    class FakeResponse:
        content = b"Password,Times\nqwerty,0.1\n"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        requests, "get", lambda *args, **kwargs: FakeResponse(), raising=False
    )

    path = tmp_path / "data.csv"

    returned_path = download_data("https://example.com/data.csv", path)

    assert returned_path == str(path)
    assert path.exists()
    assert path.read_bytes() == FakeResponse.content


def test_download_data_empty_url(tmp_path):
    with pytest.raises(ValueError):
        download_data("", tmp_path / "data.csv")


def test_download_data_http_error(monkeypatch, tmp_path):
    class FakeResponse:
        content = b""

        def raise_for_status(self):
            raise requests.HTTPError("boom")

    monkeypatch.setattr(
        requests, "get", lambda *args, **kwargs: FakeResponse(), raising=False
    )

    with pytest.raises(RuntimeError):
        download_data("https://example.com/data.csv", tmp_path / "data.csv")
