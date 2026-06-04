import os

import joblib
import uvicorn
from fastapi import Depends, FastAPI
from pydantic import BaseModel

app = FastAPI(title="FastAPI hm08")

# Получаем путь к модели из окружения или используем значение по умолчанию
model_path = os.getenv("MODEL_PATH", "pipeline.joblib")

# Кеш модели
_model = None


def get_model():
    global _model
    if _model is None:
        print("======== MODEL LOAD ========")
        print(f"Loading model from {model_path}")

        _model = joblib.load(model_path)

        try:
            pred = _model.predict(["debug"])
            print(f"Model test prediction: {pred}")
        except Exception as e:
            print("Prediction test failed:", e)

        print("======== MODEL READY ========")

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
    predictions = model.predict(request.passwords)

    # универсальное преобразование
    predictions = list(predictions)

    return PredictResponse(prediction=predictions)


# Точка входа для запуска через python app.py
def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
