import os

import pandas as pd
import numpy as np
import pytest

# Импортируем тестируемые функции
from data_utils import (
    load_parquet_chunks,
    load_dataset,
    split_dataset_by_target,
    split_target_only,
    make_file_path,
    check_data_folder_and_count_files,
    save_predictions_with_id
)

"""
Декораторо @pytest.fixture определяет фикстуру - делает результат функции доступным
для использования в тестовых функциях.
Аргумент tmp_path- 'то встроенная фикстура pytest.
При каждом запуске теста, pytest отдаёт сюда новый, пустой, изолированный путь на диске (pathlib.Path объект).
Всё, что мы туда запишем, будет удалено после тестов.
"""
@pytest.fixture
def parquet_test_dir(tmp_path):
    """
    Фикстура создаёт временную директорию с несколькими parquet-файлами
    для имитации датасета.

    Структура имён:
    train_data_0.pq, train_data_1.pq, ...

    Каждая партиция по сути будет датафреймом с колонками
    id, feature.
    """
    # генерим три файла
    for i in range(3):
        df = pd.DataFrame({
        # Генерируем 5 последовательных id,
        # которые не пересекаются между файлами.
        # В feature создаём 5 случайных чисел из нормального распределения N(0,1)
        "id": np.arange(i * 5, (i + 1) * 5),
            "feature": np.random.randn(5)
        })
        # Сохраняем датафрейм в формат Parquet
        df.to_parquet(tmp_path / f"train_data_{i}.pq")
    return tmp_path


def test_load_parquet_chunks_reads_selected_parts(parquet_test_dir):
    """
    Проверяем, что load_parquet_chunks:
    1. Читает нужное количество файлов
    2. Объединяет их в один DataFrame
    3. Работает с ограничением по колонкам
    """
    # Читаем только 2 партиции, начиная с первой
    df = load_parquet_chunks(
        path_to_dataset=parquet_test_dir,
        start_from=1,
        num_parts_to_read=2,
        verbose=False,
        columns=["id"]
    )
    # Всего 2 партиции по 5 строк = 10 строк
    assert df.shape[0] == 10
    # В колонках только 'id'
    assert list(df.columns) == ["id"]


def test_load_dataset_batches_and_saves(parquet_test_dir, tmp_path):
    """
    Проверяем, что load_dataset:
    1. Загружает данные батчами
    2. Опционально сохраняет каждый батч в файлы
    3. Возвращает полный объединенный DataFrame
    """
    save_dir = tmp_path / "processed"
    save_dir.mkdir()

    df = load_dataset(
        path_to_dataset=parquet_test_dir,
        num_parts_total=3,
        save_to_path=save_dir,
        num_parts_to_preprocess_at_once=1,
        verbose=False
    )
    # В исходных данных 3 партиции * 5 строк
    assert df.shape[0] == 15
    # Проверяем, что сохранились 3 файла
    saved_files = list(save_dir.glob("processed_chunk_*.parquet"))
    assert len(saved_files) == 3


def test_split_dataset_by_target(tmp_path):
    """
    Проверяем split_dataset_by_target:
    - создаём датасет и CSV-файл с таргетом и id
    - проверяем, что сплит с учётом стратификации возвращает непересекающиеся id
    """
    # X — набор фичей по id
    df_features = pd.DataFrame({
        "id": np.arange(10),
        "f1": np.random.randn(10)
    })

    # таргет с равным числом классов
    target_df = pd.DataFrame({
        "id": np.arange(10),
        "target": [0, 1] * 5
    })
    target_csv = tmp_path / "target.csv"
    target_df.to_csv(target_csv, index=False)

    result = split_dataset_by_target(
        dataset=df_features,
        path_to_target=target_csv,
        train_size=0.8,
        random_state=42,
        stratify_col="target"
    )

    # Проверим, что в train и test нет пересечений по id
    # set преобразует в множество тренировочный набор id.
    # isdisjoint преобразует тестовый набор id в множество
    # и возвращает True, если множества A и B не имеют общих элементов.
    assert set(result["X_train"]["id"]).isdisjoint(result["X_test"]["id"])
    # Проверим, что длины соответствуют train_size
    assert len(result["X_train"]) == 8
    assert len(result["X_test"]) == 2


def test_split_target_only(tmp_path):
    """
    Проверяем split_target_only:
    - разделяет только Series с таргетом
    - размеры соответствуют train_size
    """
    df_target = pd.DataFrame({
        "id": np.arange(6),
        "target": [0, 1] * 3
    })
    csv_path = tmp_path / "target.csv"
    df_target.to_csv(csv_path, index=False)

    split_res = split_target_only(
        path_to_target=csv_path,
        train_size=0.5,
        random_state=42,
        stratify_col="target",
        verbose=False
    )

    assert len(split_res["y_train"]) == 3
    assert len(split_res["y_test"]) == 3


def test_make_file_path_creates_expected_name():
    """
    Проверка make_file_path:
    - имя файла содержит output_type, имя базовой папки и расширение
    - путь корректно собирается в output_dir
    """
    output_path = make_file_path(
        output_type="predict",
        data_path="data/raw",
        output_dir="predictions/inference",
        ext="csv"
    )
    # Имя файла должно содержать predict__raw__ и .csv
    assert "predict__raw__" in os.path.basename(output_path)
    assert output_path.endswith(".csv")


def test_check_data_folder_and_count_files(tmp_path):
    """
    Проверка check_data_folder_and_count_files:
    - создаётся несколько файлов по маске
    - функция находит все и считает их количество
    """
    for i in range(3):
        (tmp_path / f"file_{i}.txt").write_text("test")

    files, count = check_data_folder_and_count_files(
        data_path=tmp_path,
        pattern="*.txt"
    )
    assert count == 3
    assert all(os.path.splitext(f)[1] == ".txt" for f in files)

def test_check_data_folder_and_count_files_raises(tmp_path):
    """
    Проверка, что при отсутствии файлов по маске возбуждается ValueError
    """
    # Если код внутри with не вызовет ValueError, то тест упадёт
    # Если вызовет другой тип исключения, то тест тоже упадёт
    # Только если действительно будет выброшен ValueError, тест пройдёт
    with pytest.raises(ValueError):
        check_data_folder_and_count_files(
            data_path=tmp_path,
            pattern="*.csv"
        )


def test_save_predictions_with_id_proba_and_predict(tmp_path):
    """
    Проверка save_predictions_with_id:
    - создаём id и предсказания для proba и для predict
    - проверяем структуру сохранённого csv
    """
    ids = [1, 2, 3]
    proba = np.array([[0.2, 0.8],
                      [0.6, 0.4],
                      [0.1, 0.9]])

    # Сохраняем вероятности
    proba_path = tmp_path / "proba.csv"
    save_predictions_with_id("proba", ids, proba, proba_path)
    df1 = pd.read_csv(proba_path)
    assert list(df1.columns) == ["id", "proba_class_0", "proba_class_1"]

    # Сохраняем метки
    labels = np.array([0, 1, 1])
    pred_path = tmp_path / "pred.csv"
    save_predictions_with_id("predict", ids, labels, pred_path)
    df2 = pd.read_csv(pred_path)
    assert list(df2.columns) == ["id", "prediction"]
