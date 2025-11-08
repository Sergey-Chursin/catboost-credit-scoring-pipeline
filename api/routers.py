import pandas as pd
from fastapi import APIRouter, HTTPException
from src.config import PIPELINE_PATH, THRESHOLD
from src.log_config import setup_logging
from src.pipeline import load_pipeline

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    SinglePrediction,
)

router = APIRouter()

# Чтобы не вызывать пайплайн второй раз для метода predict
# получим метки классов из их вероятностей используя вычисленный порог
threshold: float = THRESHOLD

# Настроим логер и передадим его в пайплайн
logger = setup_logging("info")

# Загружаем обученный пайплайн
pipeline = load_pipeline(PIPELINE_PATH, logger)


@router.get("/health")
async def health():
    """
    Эндпоинт для мониторинга Доккером доступности сервиса и автоматического перезапуска.
    Параметры мониторинга настроены в Docker-compose
    """
    return {"status": "ok", "pipeline_loaded": pipeline is not None}


# Декоратор регистрирует функцию на POST-запросы к адресу /predict,
# Аргумент response_model=PredictionResponse заставляет FastAPI привести результат функции
# к заданной Pydantic-схеме — это гарантирует структуру JSON-ответа
@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Эндпоинт для получения вероятности дефолта по данным клиента.
    На вход: объект PredictionRequest с данными по клиенту.
    Запрос ожидает тело в формате, описанном схемой PredictionRequest.
    На выход: JSON с вероятностью и меткой класса.
    """
    # Проверяем наличие пайплайна
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    # преобразуем каждую модель в dict
    records = [item.model_dump() for item in request.data]

    # Конвертируем список словарей в DataFrame удобный для sklearn
    df = pd.DataFrame.from_records(records)

    # Получаем предикты
    proba = pipeline.predict_proba(df)[:, 1]
    pred = proba >= threshold

    # Собираем список словарей по предиктам клиентов
    # Кастуем типы предиктов так json не поддерживает numpy float 64 и int64
    # возвращаемые пайплайном.
    # df["id"].unique() - обрабатывает случаи с несколькими записями для одного клиента
    resp = [
        SinglePrediction(
            client_id=int(unique_id),
            probability=float(prob),
            class_label=int(label),
        )
        for unique_id, prob, label in zip(
            df["id"].unique(),
            proba,
            pred,
            strict=True,
        )
    ]
    # Создаём экземпляр схемы-ответа, автоматически сериализуем в JSON
    return PredictionResponse(predictions=resp)
