from fastapi.testclient import TestClient
from app.main import app
from app.model_loader import set_model_for_tests


class FakeModel:
    def predict(self, values):
        return [float(len(x)) for x in values]


client = TestClient(app)


def test_predict_success():
    set_model_for_tests(FakeModel())

    response = client.post("/predict", json={"Password": ["qwerty", "123456"]})

    assert response.status_code == 200
    assert response.json() == {"Times": [6.0, 6.0]}


def test_predict_invalid_payload():
    set_model_for_tests(FakeModel())

    response = client.post("/predict", json={"passwords": ["qwerty"]})

    assert response.status_code == 422


def test_predict_empty_password_list():
    set_model_for_tests(FakeModel())

    response = client.post("/predict", json={"Password": []})

    assert response.status_code == 422
