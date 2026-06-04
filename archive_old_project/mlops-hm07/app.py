import os

import joblib
import uvicorn
from fastapi import Depends, FastAPI
from pydantic import BaseModel

app = FastAPI(title="FastAPI hm07")

# Получаем путь к модели из окружения или используем значение по умолчанию
model_path = os.getenv("MODEL_PATH", "pipeline.joblib")

# Кеш модели
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(model_path)
    return _model


# Схема запроса
class PredictRequest(BaseModel):
    passwords: list[str]


# Схема ответа
class PredictResponse(BaseModel):
    prediction: list[float]


# Эндпоинт
@app.post("/predict")
def predict(request: PredictRequest, model=Depends(get_model)) -> PredictResponse:
    predictions = model.predict(
        request.passwords
    )  # Получаем предсказания для списка паролей
    return PredictResponse(
        prediction=predictions.tolist()
    )  # Преобразуем numpy-массив в список Python


# Точка входа для запуска через python app.py
def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
