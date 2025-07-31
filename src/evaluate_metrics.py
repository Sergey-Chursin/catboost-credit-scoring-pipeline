import logging
from typing import Sequence
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    TARGET_PATH,
    TRAIN_SIZE,
    SEED_SPLIT_DATASET,
    STRATIFY_COL,
    PROBA_TEST_PREDICT,
    CLASSES_TEST_PREDICT
)

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score
)

import pickle

"""
Создаём локальный логгер для этого модуля
Он наследует настройки от root logger
импортирующего файла (pipeline.py)
"""
logger = logging.getLogger(__name__)
verbose = True


def split_target_only(
        path_to_target: str = TARGET_PATH,
        train_size: float = TRAIN_SIZE,
        random_state: int = SEED_SPLIT_DATASET,
        stratify_col: str = STRATIFY_COL,
        verbose: bool = True
):
    """
    Разделяет только таргет на train/test подвыборки.

    Args:
        По умолчанию все аргументы берутся из config.
        path_to_target: путь к target.csv
        train_size: доля train
        random_state: seed для воспроизводимости
        stratify_col: по какой колонке стратифицироваться
        verbose: Вывод print. По умолчанию True.
    Returns:
        dict с pandas.Series: {'y_train', 'y_test'}
    """
    target = pd.read_csv(path_to_target)
    if verbose:
        print(f'Loaded target from {path_to_target}'
              f' (shape: {target.shape}'
              )

    y_train, y_test = train_test_split(
        target,
        train_size=train_size,
        random_state=random_state,
        stratify=target[stratify_col]
    )
    if verbose:
        print(
            f'y_train shape: {y_train.shape}\n'
            f'y_test shape: {y_test.shape}'
        )
    return {
        'y_train': y_train[stratify_col],
        'y_test': y_test[stratify_col]
    }

def evaluate_auc_score(
        y_true: Sequence,
        y_score: Sequence,
        verbose: bool = True
) -> float:
    """
    Вычисляет и выводит/логирует значение ROC AUC для тестовой выборки.

    Args:
        y_true (Sequence): Истинные метки классов (обычно 1D array, pandas.Series или список).
        y_score (Sequence): Предсказанные вероятности
            (обычно 1D array, pandas.Series или список вероятностей класса 1).
        verbose (bool, optional): Если True, печатает AUC в консоль. По умолчанию True.

    Returns:
        float: Значение метрики ROC AUC на тестовой выборке.
    """
    # Если подали двумерный массив (n_samples, 2) — берём вероятности для класса 1
    y_score = np.asarray(y_score)
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        y_score = y_score[:, 1]

    # Считаем метрику
    auc = roc_auc_score(y_true, y_score)

    # Выводим значение метрики
    logger.info(f"AUC on test set: {auc:.4f}")
    if verbose:
        print(f"AUC on test set: {auc:.4f}")
    return auc

def evaluate_accuracy_score(
        y_true: Sequence,
        y_pred: Sequence,
        verbose: bool = True
) -> float:
    """
    Вычисляет и выводит/логирует значение Accuracy для тестовой выборки.

    Args:
        y_true (Sequence): Истинные метки классов (обычно 1D array, pandas.Series или список).
        y_pred (Sequence): Предсказанные классы (обычно 1D array, pandas.Series или список меток классов).
        verbose (bool, optional): Если True, печатает Accuracy в консоль. По умолчанию True.

    Returns:
        float: Значение метрики Accuracy на тестовой выборке.
    """
    acc = accuracy_score(y_true, y_pred)
    logger.info(f"Accuracy on test set: {acc:.4f}")
    if verbose:
        print(f"Accuracy on test set: {acc:.4f}")
    return acc


if __name__ == "__main__":
    y_dict = split_target_only()
    y_true = y_dict['y_test']

    # Загружаем предикты вероятностей классов
    with open(PROBA_TEST_PREDICT, 'rb') as f:
        probabilities = pickle.load(f)

    if verbose:
        print(
            f'Loaded predicted probabilities from {PROBA_TEST_PREDICT}\n'
            f' shape: {probabilities.shape}'
        )

    # Загружаем предикты классов
    with open(CLASSES_TEST_PREDICT, 'rb') as f:
        classes = pickle.load(f)

    if verbose:
        print(
            f'Loaded predicted classes from {CLASSES_TEST_PREDICT}\n'
            f' shape: {classes.shape}'
        )
    # Проверяем совпадение длинн предикта и таргета -
    # выбрасываем предупреждение если нет.
    assert len(y_true) == len(classes), "Длины y_true и classes не совпадают!"

    # получаем вероятности класса 1
    y_score = probabilities[:, 1]
    # Проверяем совпадение длинн предикта и таргета -
    # выбрасываем предупреждение если нет.
    assert len(y_true) == len(y_score), "Длины y_true и y_score не совпадают!"

    # Вызываем функцию оценки AUC
    evaluate_auc_score(
            y_true,
            y_score,
    )

    # Вызываем функцию оценки accuracy
    evaluate_accuracy_score(
        y_true,
        classes,
    )