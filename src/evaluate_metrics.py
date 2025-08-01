import logging
from typing import Sequence
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    TARGET_PATH,
    TRAIN_SIZE,
    SEED_SPLIT_DATASET,
    STRATIFY_COL,
    PROBA_TEST_PREDICT,
    CLASSES_TEST_PREDICT,
    CLASSES_METRIC_LIST
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

def pred_and_metrics_compatible(
        y_pred: np.ndarray,
        eval_metric: str
) -> bool:
    """
    Вспомогательная функция для compute_and_log_metrics.
    Проверяет, соответствует ли тип y_pred требованиям конкретной метрики.
    Возвращает True, если да (и можно использовать этот предикт для расчёта метрики), иначе False.
    """
    if eval_metric in CLASSES_METRIC_LIST:
        # Метрики по меткам классов (accuracy, f1, ...): одномерный вектор целых чисел
        # isinstance(y_pred, np.ndarray) -проверка на массив
        # y_pred.ndim - проверка на рвзмерность
        # np.issubdtype - проверка на тип
        return (
                isinstance(y_pred, np.ndarray)
                and y_pred.ndim == 1
                and np.issubdtype(y_pred.dtype, np.integer)
        )
    else:
        # Метрики по вероятностям (auc, logloss, ...) — (n,2) float или (n,) float
        # Добавлен вариант с predict_proba прошедшей слайсинг, то есть с одномерным
        # массивом float
        return (
                isinstance(y_pred, np.ndarray)
                and (
                        (y_pred.ndim == 2 and y_pred.shape[1] == 2 and np.issubdtype(y_pred.dtype, np.floating))
                        or (y_pred.ndim == 1 and np.issubdtype(y_pred.dtype, np.floating))
                )
        )

def compute_and_log_metrics(
    eval_metric: str,
    pipe: Any,
    train_test_dict: Dict[str, pd.DataFrame],
    y_pred: Optional[np.ndarray] = None
) -> Optional[float]:
    """
    Вычисляет и логирует выбранную метрику качества (AUC или Accuracy) на тестовой выборке.

    Args:
        eval_metric (str): Краткое имя метрики ('auc', 'acc', 'off').
        pipe (Any): Обученный пайплайн.
        train_test_dict (dict): Словарь с тестовыми данными, должен содержать ключи
            'X_test' (pd.DataFrame) и 'y_test' (pd.Series или 1D np.array).
        y_pred (np.ndarray, optional): заранее полученный предикт,
             используется если совместим с eval_metric.

    Returns:
        Optional[float]: Значение метрики (ROC AUC или Accuracy) на тестовой выборке,
            либо None, если выбран режим 'off' или флаг не введён.
    """
    logger.info("Function compute_and_log_metrics started")
    # Словарь для маппинг диспетчеризации
    eval_metrics_map = {
        'auc': evaluate_auc_score,
        'acc': evaluate_accuracy_score
    }
    # Выбираем функцию из словаря по аргументу eval_metric
    func = eval_metrics_map.get(eval_metric)

    # В случае off или отсутствия флага
    if not func:
        logger.info("No evaluation metric selected (off mode).")
        return None
    # Получаем тестовые данные из словаря
    X_test = train_test_dict['X_test']
    y_test = train_test_dict['y_test']

    # Если подан y_pred нужного формата — используем его
    # Проверка размерности предикта есть в функциях модуля evaluate_metrics
    # verbose=False для отключения print() в функциях модуля evaluate_metrics
    if y_pred is not None and pred_and_metrics_compatible(y_pred, eval_metric):
        logger.info(f"Using provided y_pred for metric '{eval_metric}'")
        result = func(y_test, y_pred, verbose=False)

    else:
        # Иначе делаем свежий инференс подходящего типа через pipeline
        # Если для метрики нужны метки классов
        if eval_metric in CLASSES_METRIC_LIST:
            logger.info(f"Calculating {eval_metric.upper()}: performing predict")
            # Делаем предикт
            y_pred = pipe.predict(X_test)
            # verbose=False для отключения print() в функциях модуля evaluate_metrics
            result = func(y_test, y_pred, verbose=False)
        else:
            # В остальных случаях делаем predict_proba
            logger.info(f"Calculating {eval_metric.upper()}: performing predict_proba")
            y_pred = pipe.predict_proba(X_test)[:, 1]
            # verbose=False для отключения print() в функциях модуля evaluate_metrics
            result = func(y_test, y_pred, verbose=False)

    return result


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