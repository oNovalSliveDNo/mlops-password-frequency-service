import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_predict(client: TestClient):
    response = client.post(
        "/predict", json={"passwords": ["qwerty123", "password", "admin"]}
    )
    assert response.status_code == 200
