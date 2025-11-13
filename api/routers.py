import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from src.constants import PIPELINE_PATH, THRESHOLD
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


@router.get(
    "/",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def root():
    """
    Корневой эндпоинт с HTML landing page
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Credit Scoring API</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #333;
            }
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 50px;
                max-width: 700px;
                width: 90%;
            }
            h1 {
                color: #667eea;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                color: #666;
                margin-bottom: 40px;
                font-size: 1.1em;
            }
            .section {
                margin-bottom: 30px;
            }
            .section h2 {
                color: #444;
                margin-bottom: 15px;
                font-size: 1.5em;
            }
            .links {
                list-style: none;
            }
            .links li {
                margin: 12px 0;
            }
            .links a {
                display: flex;
                align-items: center;
                padding: 15px 20px;
                background: #f8f9fa;
                border-radius: 10px;
                text-decoration: none;
                color: #667eea;
                transition: all 0.3s ease;
                font-weight: 500;
            }
            .links a:hover {
                background: #667eea;
                color: white;
                transform: translateX(5px);
            }
            .emoji {
                font-size: 1.5em;
                margin-right: 15px;
            }
            .footer {
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                text-align: center;
                color: #999;
                font-size: 0.9em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Credit Scoring API</h1>
            <p class="subtitle">CatBoost Ensemble-based Credit Scoring System</p>

            <div class="section">
                <h2>📚 Documentation</h2>
                <ul class="links">
                    <li>
                        <a href="/docs">
                            <span class="emoji">📖</span>
                            <span>Swagger UI - Interactive API Documentation</span>
                        </a>
                    </li>
                    <li>
                        <a href="/redoc">
                            <span class="emoji">📄</span>
                            <span>ReDoc - Alternative Documentation View</span>
                        </a>
                    </li>
                </ul>
            </div>

            <div class="section">
                <h2>🔗 API Endpoints</h2>
                <ul class="links">
                    <li>
                        <a href="/health">
                            <span class="emoji">✅</span>
                            <span>Health Check - API Status</span>
                        </a>
                    </li>
                    <li>
                        <a href="/docs#/default/predict_predict_post">
                            <span class="emoji">🎲</span>
                            <span>Predict - Make Credit Score Predictions</span>
                        </a>
                    </li>
                </ul>
            </div>

            <div class="footer">
                <p>Built with FastAPI | CatBoost | Python</p>
            </div>
        </div>
    </body>
    </html>
    """


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
