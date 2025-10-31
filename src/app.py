"""
FastAPI-сервис для получения предиктов кредитного скоринга.

Поддерживает как индивидуальную обработку данных одного клиента,
так и batch‑скоринг (список клиентов с набором признаков).
Для одного клиента можно подать одну или несколько записей описывающих
кредиты клиента, требуется только объединить их одним ID.

Все объекты batch-запроса автоматически проверяются на структуру и типы признаков
по схеме FeatureVector (pydantic), что обеспечивает безопасность,
воспроизводимость и предсказуемость обработки.
Некорректные записи отклоняются с подробным описанием ошибки.

На выходе получается агрегированный предикт для каждого клиента,
 включающий ID клиента, вероятность дефолта и бинарную метку предсказания
(0 — дефолт не ожидается, 1 — дефолт ожидается).

Предусмотрен endpoint для проверки статуса пайплайна - health check,
в Docker Compose на его основе настроен автоматический перезапуск сервиса.

Endpoints:
/predict (POST): принимает batch-запрос, возвращает предсказания.
/health (GET): проверка готовности и загрузки пайплайна.

Response format:
predictions: список словарей {client_id, probability, class}
model_info: строка с краткой информацией о модели.

Пример запроса и ответа приведены в openapi schema.

Документация OpenAPI доступна по относительным адресам
/docs (Swagger UI) и /redoc (Redoc) на том хосте и порту, где запущен сервис.
Пример: http://localhost:8000/docs
В Swagger UI можно не только ознакомиться с описанием, но и отправлять тестовые
запросы к любому endpoint прямо из браузера.
"""

from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import PIPELINE_PATH, THRESHOLD
from src.log_config import setup_logging
from src.pipeline import load_pipeline

# Создадим объект FastAPI
app = FastAPI(title="Credit-Scoring API", version="1.0.0")

# Чтобы не вызывать пайплайн второй раз для метода predict
# получим метки классов из их вероятностей используя вычисленный порог
threshold = THRESHOLD

# Настроим логер и передадим его в пайплайн
logger = setup_logging("info")

# Загружаем обученный пайплайн
pipeline = load_pipeline(PIPELINE_PATH, logger)


# Создаём схему запроса
# Определяем вложенную модель (структуру признаков одного объекта)
class FeatureVector(BaseModel):
    id: int
    rn: int
    pre_since_opened: int
    pre_since_confirmed: int
    pre_pterm: int
    pre_fterm: int
    pre_till_pclose: int
    pre_till_fclose: int
    pre_loans_credit_limit: int
    pre_loans_next_pay_summ: int
    pre_loans_outstanding: int
    pre_loans_max_overdue_sum: int
    pre_loans_credit_cost_rate: int
    pre_loans5: int
    pre_loans530: int
    is_zero_loans5: int
    is_zero_loans530: int
    pre_util: int
    pre_over2limit: int
    is_zero_over2limit: int
    enc_paym_0: int
    enc_paym_1: int
    enc_paym_2: int
    enc_paym_8: int
    enc_paym_9: int
    enc_paym_10: int
    enc_paym_24: int
    enc_loans_account_holder_type: int
    enc_loans_credit_status: int
    enc_loans_credit_type: int
    enc_loans_account_cur: int
    is_zero_loans3060: int
    is_zero_loans6090: int
    is_zero_loans90: int
    enc_paym_3: int
    enc_paym_4: int
    enc_paym_5: int
    enc_paym_6: int
    enc_paym_7: int
    enc_paym_11: int
    enc_paym_12: int
    enc_paym_13: int
    enc_paym_14: int
    enc_paym_15: int
    enc_paym_16: int
    enc_paym_17: int
    enc_paym_18: int
    enc_paym_19: int
    enc_paym_20: int
    enc_paym_21: int
    enc_paym_22: int
    enc_paym_23: int


# Второй моделью делаем запрос – список объектов FeatureVector
# Запрос - это словарь с одним ключом 'data' и списком словарей внутри
class PredictionRequest(BaseModel):
    data: List[FeatureVector]


# Модель ответа - список словарей(1 словарь = 1 клиент) и информация о модели
class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_info: str = "CatBoost ensemble, 6 models, binary classification"


@app.get("/health")
async def health():
    """
    Эндпоинт для мониторинга Доккером доступности сервиса и автоматического перезапуска.
    Параметры мониторинга настроены в Docker-compose
    """
    return {"status": "ok", "pipeline_loaded": pipeline is not None}


# Декоратор регистрирует функцию на POST-запросы к адресу /predict,
# Аргумент response_model=PredictionResponse заставляет FastAPI привести результат функции
# к заданной Pydantic-схеме — это гарантирует структуру JSON-ответа
@app.post("/predict", response_model=PredictionResponse)
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
        {"client_id": int(unique_id), "probability": float(prob), "class": int(label)}
        for unique_id, prob, label in zip(df["id"].unique(), proba, pred, strict=True)
    ]
    # Создаём экземпляр схемы-ответа, автоматически сериализуем в JSON
    return PredictionResponse(predictions=resp)


# Локальный запуск для проверки
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", host="0.0.0.0", port=8000)
