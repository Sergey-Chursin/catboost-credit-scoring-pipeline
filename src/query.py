import os

import pandas as pd

pd.set_option("display.max_columns", None)
import json

import requests

from src.config import PRE_FEATURES, RAW_DATA_PATH, TRANSFORM_DATA_PATH
from src.data_utils import load_dataset

if __name__ == "__main__":
    data = load_dataset(RAW_DATA_PATH, num_parts_total=1, columns=PRE_FEATURES)
    print("data.shape:", data.shape)

    # СОБЕРЁМ ЗАПРОС С ОДНИМ КЛИЕНТОМ С ОДНИМ КРЕДИТОМ
    print("ONE CLIENT WITH ONE LOAN")
    print(data["id"].value_counts().tail(1))

    one_id_one_loan = data[data["id"] == 125000]
    print("one_id_one_loan.shape:", one_id_one_loan.shape)

    one_id_one_loan = one_id_one_loan.to_json(orient="records")
    one_id_one_loan = {"data": json.loads(one_id_one_loan)}

    # Сохраним запрос
    filename = os.path.join(TRANSFORM_DATA_PATH, "one_id_one_loan.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(one_id_one_loan, f, ensure_ascii=False)

    response = requests.post(url="http://localhost:8000/predict", json=one_id_one_loan)
    print("response.status_code:", response.status_code)
    print("response.json:", response.json())

    # CОБЕРЁМ ЗАПРОС С ОДНИМ КЛИЕНТОМ С НЕСКОЛЬКИМИ КРЕДИТАМИ
    print("ONE CLIENT WITH SEVERAL LOANS")

    one_id_many_loans = data[data["id"] == 0]
    print("one_id_many_loans.shape:", one_id_many_loans.shape)

    one_id_many_loans = one_id_many_loans.to_json(orient="records")
    one_id_many_loans = {"data": json.loads(one_id_many_loans)}

    # Сохраним запрос
    filename = os.path.join(TRANSFORM_DATA_PATH, "one_id_many_loans.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(one_id_many_loans, f, ensure_ascii=False)

    response = requests.post(
        url="http://localhost:8000/predict", json=one_id_many_loans
    )
    print("response.status_code:", response.status_code)
    print("response.json:", response.json())

    # CОБЕРЁМ ЗАПРОС С НЕСКОЛЬКИМИ КЛИЕНТАМИ С ОДНИМ КРЕДИТОМ
    print("SEVERAL CLIENTS EACH HAVE ONE LOAN")
    print(data["id"].value_counts().tail(5))

    many_id_one_loan = data[
        (data["id"] == 125000)
        | (data["id"] == 97643)
        | (data["id"] == 220124)
        | (data["id"] == 97668)
        | (data["id"] == 220126)
    ]
    print("many_id_one_loan.shape:", many_id_one_loan.shape)

    many_id_one_loan = many_id_one_loan.to_json(orient="records")
    many_id_one_loan = {"data": json.loads(many_id_one_loan)}

    # Сохраним запрос
    filename = os.path.join(TRANSFORM_DATA_PATH, "many_id_one_loan.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(many_id_one_loan, f, ensure_ascii=False)

    response = requests.post(url="http://localhost:8000/predict", json=many_id_one_loan)
    print("response.status_code:", response.status_code)
    print("response.json:", response.json())

    # CОБЕРЁМ ЗАПРОС С НЕСКОЛЬКИМИ КЛИЕНТАМИ С НЕСКОЛЬКИМИ КРЕДИТАМИ
    print("SEVERAL CLIENTS EACH HAVE SEVERAL LOANS")
    print(data["id"].value_counts().head(4))

    many_id_many_loans = data[
        (data["id"] == 0)
        | (data["id"] == 62963)
        | (data["id"] == 77107)
        | (data["id"] == 72660)
        | (data["id"] == 4951)
    ]
    print("many_id_many_loans.shape:", many_id_many_loans.shape)

    many_id_many_loans = many_id_many_loans.to_json(orient="records")
    many_id_many_loans = {"data": json.loads(many_id_many_loans)}

    # Сохраним запрос
    filename = os.path.join(TRANSFORM_DATA_PATH, "many_id_many_loans.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(many_id_many_loans, f, ensure_ascii=False)

    response = requests.post(
        url="http://localhost:8000/predict", json=many_id_many_loans
    )
    print("response.status_code:", response.status_code)
    print("response.json:", response.json())

    # СРАВНИМ РЕЗУЛЬТАТЫ С РЕЗУЛЬТАТАМИ ОБЩЕГО ПРЕДИКТА ПОЛУЧЕННОГО В INFERENCE РЕЖИМЕ
    """
     Использовалось для проверки воспроизводимости, но так как сохранять огромный файл
    референса я не хочу, а при новом запуске он поменяет название, то эту часть я закрою.
    """
    # predict = pd.read_csv("../predictions/inference/predict_raw_2025-10-02-16-34.csv")
    # print("predict.shape:", predict.shape)
    #
    # predict = predict[
    #     (predict["id"] == 0)
    #     | (predict["id"] == 125000)
    #     | (predict["id"] == 220126)
    #     | (predict["id"] == 97643)
    #     | (predict["id"] == 220124)
    #     | (predict["id"] == 97668)
    #     | (predict["id"] == 62963)
    #     | (predict["id"] == 77107)
    #     | (predict["id"] == 72660)
    #     | (predict["id"] == 4951)
    # ]
    # print(predict)
    # #   0   2531540 2349517 1108612 883552 1903673  2891734 1782755 1658519 62963
    #
    # proba = pd.read_csv("../predictions/inference/proba_raw_2025-10-02-16-23.csv")
    # print("proba.shape:", proba.shape)
    #
    # proba = proba[
    #     (proba["id"] == 0)
    #     | (proba["id"] == 125000)
    #     | (proba["id"] == 220126)
    #     | (proba["id"] == 97643)
    #     | (proba["id"] == 220124)
    #     | (proba["id"] == 97668)
    #     | (proba["id"] == 62963)
    #     | (proba["id"] == 77107)
    #     | (proba["id"] == 72660)
    #     | (proba["id"] == 4951)
    # ]
    #
    # print(proba)
