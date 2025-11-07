import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.data_utils import (
    check_data_folder_and_count_files,
    load_data_chunks,
    load_dataset,
    make_file_path,
    save_predictions_with_id,
    split_dataset_by_target,
)

"""
tmp_path — это встроенная фикстура pytest, которая возвращает объект типа pathlib.Path. 
Это временная директория, уникальная для каждого теста, 
которая автоматически удаляется после завершения теста.
"""


def test_load_parquet_chunks_reads_selected_parts(
    parquet_test_dir: Path,
) -> None:
    """
    Проверяем, что load_data_chunks:
    1. Читает нужное количество файлов
    2. Объединяет их в один DataFrame
    3. Работает с ограничением по колонкам
    """
    # Читаем только 2 партиции, начиная с первой
    df = load_data_chunks(
        path_to_dataset=str(parquet_test_dir),
        start_from=1,
        num_parts_to_read=2,
        verbose=False,
        columns=["id"],
    )
    # Всего 2 партиции по 5 строк = 10 строк
    assert df.shape[0] == 10
    # В колонках только 'id'
    assert list(df.columns) == ["id"]


def test_load_dataset_batches_and_saves(
    parquet_test_dir: Path,
    tmp_path: Path,
) -> None:
    """
    Проверяем, что load_dataset:
    1. Загружает данные батчами
    2. Опционально сохраняет каждый батч в файлы
    3. Возвращает полный объединенный DataFrame
    """
    save_dir = tmp_path / "processed"
    save_dir.mkdir()

    df = load_dataset(
        path_to_dataset=str(parquet_test_dir),
        num_parts_total=3,
        save_to_path=str(save_dir),
        num_parts_to_preprocess_at_once=1,
        verbose=False,
    )
    assert df is not None
    # В исходных данных 3 партиции * 5 строк
    assert df.shape[0] == 15
    # Проверяем, что сохранились 3 файла
    saved_files = list(save_dir.glob("processed_chunk_*.parquet"))
    assert len(saved_files) == 3


def test_split_dataset_by_target(tmp_path: Path) -> None:
    """
    Проверяем split_dataset_by_target:
    - создаём датасет и CSV-файл с таргетом и id
    - проверяем, что сплит с учётом стратификации возвращает непересекающиеся id
    """
    df_features = pd.DataFrame(
        {
            "id": np.arange(10),
            "f1": np.random.randn(10),
        }
    )
    # таргет с равным числом меток классов
    target_df = pd.DataFrame(
        {
            "id": np.arange(10),
            "target": [0, 1] * 5,
        }
    )
    target_csv = tmp_path / "target.csv"
    target_df.to_csv(target_csv, index=False)

    result = split_dataset_by_target(
        dataset=df_features,
        path_to_target=str(target_csv),
        train_size=0.8,
        random_state=42,
        stratify_col="target",
    )

    # Проверим, что в train и test нет пересечений по id
    # isdisjoint() возвращает True, если множества не имеют общих элементов
    train_ids = set(result["X_train"]["id"].tolist())
    test_ids = set(result["X_test"]["id"].tolist())
    assert train_ids.isdisjoint(test_ids)

    # Проверим, что длины соответствуют train_size
    assert len(result["X_train"]) == 8
    assert len(result["X_test"]) == 2


def test_make_file_path_creates_expected_name() -> None:
    """
    Проверка make_file_path:
    - имя файла содержит output_type, имя базовой папки и расширение
    - путь корректно собирается в output_dir
    """
    output_path = make_file_path(
        output_type="predict",
        data_path="data/raw",
        output_dir="predictions/inference",
        ext="csv",
    )
    # Имя файла должно содержать predict__raw__ и .csv
    assert "predict_raw_" in os.path.basename(output_path)
    assert output_path.endswith(".csv")


def test_check_data_folder_and_count_files(tmp_path: Path) -> None:
    """
    Проверка check_data_folder_and_count_files:
    - создаётся несколько файлов по маске
    - функция находит все и считает их количество
    """
    for i in range(3):
        (tmp_path / f"file_{i}.txt").write_text("test")

    files, count = check_data_folder_and_count_files(
        data_path=str(tmp_path), pattern="*.txt"
    )
    assert count == 3
    assert all(os.path.splitext(f)[1] == ".txt" for f in files)


def test_check_data_folder_and_count_files_raises(tmp_path: Path) -> None:
    """
    Проверка, что при отсутствии файлов по маске возбуждается ValueError
    """
    # Если код внутри with не вызовет ValueError или вызовет другой тип исключения,
    # то тест тоже упадёт
    with pytest.raises(ValueError):
        check_data_folder_and_count_files(
            data_path=str(tmp_path),
            pattern="*.csv",
        )


def test_save_predictions_with_id_proba_and_predict(tmp_path: Path) -> None:
    """
    Проверка save_predictions_with_id:
    - создаём id и предсказания для proba и для predict
    - проверяем структуру сохранённого csv
    """
    ids = pd.Series([1, 2, 3])
    proba = np.array(
        [
            [0.2, 0.8],
            [0.6, 0.4],
            [0.1, 0.9],
        ]
    )

    # Сохраняем вероятности
    proba_path = tmp_path / "proba.csv"
    save_predictions_with_id(
        "proba",
        ids,
        proba,
        str(proba_path),
    )
    df1 = pd.read_csv(proba_path)
    assert list(df1.columns) == [
        "id",
        "proba_class_0",
        "proba_class_1",
    ]

    # Сохраняем метки
    labels = np.array([0, 1, 1])
    pred_path = tmp_path / "pred.csv"
    save_predictions_with_id(
        "predict",
        ids,
        labels,
        str(pred_path),
    )
    df2 = pd.read_csv(pred_path)
    assert list(df2.columns) == [
        "id",
        "prediction",
    ]
