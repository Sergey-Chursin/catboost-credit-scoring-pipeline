from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.classifier import CatBoostEnsembleClassifier
from src.data_utils import SplitDataset

"""
Декораторо @pytest.fixture определяет фикстуру - делает результат функции доступным
для использования в тестовых функциях
"""


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """Сбалансированные синтетические данные для тестов (1000 сэмплов)."""
    # фиксируем "зерно" для воспроизводимости.
    np.random.seed(42)

    # по 500 объектов на класс
    n_per_class = 500
    X = pd.DataFrame(
        {
            # Признак f1:
            # - первые 500 значений: нормальное распределение со средним 0
            # - вторые 500 значений: нормальное распределение, но сдвинутое на +1 по среднему
            "f1": np.concatenate(
                [
                    np.random.randn(n_per_class),  # класс 0
                    np.random.randn(n_per_class) + 1,
                ]  # класс 1
            ),
            # Признак f2:
            # - первые 500 значений: нормальное N(0, σ=5)
            # - вторые 500 значений: N(2, σ=5) — т.е. сдвинуто ещё и среднее на +2
            "f2": np.concatenate(
                [
                    np.random.randn(n_per_class) * 5,  # класс 0
                    np.random.randn(n_per_class) * 5 + 2,
                ]  # класс 1
            ),
        }
    )
    # Таргет
    # первые 500 объектов — класс 0,
    # следующие 500 — класс 1
    y = pd.Series([0] * n_per_class + [1] * n_per_class)

    return X, y


@pytest.fixture
def clf_small() -> CatBoostEnsembleClassifier:
    """Классификатор с минимальными параметрами, чтобы тесты работали быстро."""
    return CatBoostEnsembleClassifier(
        # 5 фолдов + финальная
        params_list=[{"iterations": 1, "verbose": 0}] * 6,
        weights_list=[1] * 6,
        n_splits=5,
    )


"""
Аргумент tmp_path- 'то встроенная фикстура pytest.
При каждом запуске теста, pytest отдаёт сюда новый, пустой, изолированный путь на диске (pathlib.Path объект).
Всё, что мы туда запишем, будет удалено после тестов.
"""


@pytest.fixture
def parquet_test_dir(tmp_path: Path) -> Path:
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
        df = pd.DataFrame(
            {
                # Генерируем 5 последовательных id,
                # которые не пересекаются между файлами.
                # В feature создаём 5 случайных чисел из нормального распределения N(0,1)
                "id": np.arange(i * 5, (i + 1) * 5),
                "feature": np.random.randn(5),
            }
        )
        # Сохраняем датафрейм в формат Parquet
        df.to_parquet(tmp_path / f"train_data_{i}.pq")
    return tmp_path


@pytest.fixture
def train_test_dict() -> SplitDataset:
    train_test_dict: SplitDataset = {
        "X_train": pd.DataFrame({"x1": [1, 2, 3]}),
        "y_train": pd.Series([0, 1, 0]),
        "X_test": pd.DataFrame({"x1": [1, 2, 3]}),
        "y_test": pd.Series([0, 1, 0]),
    }
    return train_test_dict
