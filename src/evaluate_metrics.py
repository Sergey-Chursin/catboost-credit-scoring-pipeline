import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.data_utils import SplitDataset

"""
Создаём локальный логгер для этого модуля
Настройки (уровень логирования, формат сообщений) наследуются от root logger, 
который обычно конфигурируется в главном файле проекта (pipeline.py).
"""
logger = logging.getLogger(__name__)


def evaluate_auc_score(
    y_true: pd.Series,
    y_pred: np.ndarray,
    verbose: bool = True,
) -> float:
    """
    Вычисляет и выводит/логирует значение ROC AUC для тестовой выборки.

    Args:
        y_true (pd.Series): Истинные метки классов.
        y_pred (np.ndarray): Предсказанные вероятности.
        verbose (bool): Если True, печатает AUC в консоль. По умолчанию True.

    Returns:
        float: Значение метрики ROC AUC на тестовой выборке.
    """

    # Если подали двумерный массив (n_samples, 2) — берём вероятности для класса 1
    if y_pred.ndim == 2 and y_pred.shape[1] == 2:
        y_pred = y_pred[:, 1]

    # Считаем метрику
    auc = roc_auc_score(y_true, y_pred)

    # Выводим значение метрики
    logger.info(f"AUC on test set: {auc:.4f}")
    if verbose:
        print(f"AUC on test set: {auc:.4f}")
    return float(auc)


def evaluate_accuracy_score(
    y_true: pd.Series,
    y_pred: np.ndarray,
    verbose: bool = True,
) -> float:
    """
    Вычисляет и выводит/логирует значение Accuracy для тестовой выборки.

    Args:
        y_true (pd.Series): Истинные метки классов.
        y_pred (np.ndarray): Предсказанные классы.
        verbose (bool): Если True, печатает Accuracy в консоль. По умолчанию True.

    Returns:
        float: Значение метрики Accuracy на тестовой выборке.
    """

    acc = accuracy_score(y_true, y_pred)
    logger.info(f"Accuracy on test set: {acc:.4f}")
    if verbose:
        print(f"Accuracy on test set: {acc:.4f}")
    return float(acc)


def pred_and_metrics_compatible(
    y_pred: np.ndarray,
    eval_metric: str,
    classes_metric_list: list[str],
) -> bool:
    """
    Вспомогательная функция для compute_and_log_metrics.
    Проверяет, соответствует ли тип и размерность массива y_pred требованиям выбранной метрики.

    Для метрик, требующих метки классов (например, accuracy, f1), проверяет,
    что y_pred — это одномерный массив целых чисел.
    Для метрик по вероятностям (например, auc, logloss), проверяет,
    что y_pred либо двумерный массив float с формой (n, 2),
    либо одномерный массив float.

    Аргументы:
        y_pred (np.ndarray): Массив предсказаний (метки классов или вероятности).
        eval_metric (str): Имя метрики (например, 'acc', 'auc').
        classes_metric_list (list[str]): Список метрик, требующих метки классов.

    Возвращает:
        bool: True, если y_pred совместим с указанной метрикой; False — иначе.
    """
    if eval_metric in classes_metric_list:
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
        return isinstance(y_pred, np.ndarray) and (
            (
                y_pred.ndim == 2
                and y_pred.shape[1] == 2
                and np.issubdtype(y_pred.dtype, np.floating)
            )
            or (y_pred.ndim == 1 and np.issubdtype(y_pred.dtype, np.floating))
        )


def compute_and_log_metrics(
    eval_metric: str,
    pipe: Pipeline,
    train_test_dict: SplitDataset,
    classes_metric_list: list[str],
    y_pred: np.ndarray | None = None,
) -> float | None:
    """
    Вычисляет и логирует выбранную метрику качества (AUC или Accuracy) на тестовой выборке.

    Args:
        eval_metric (str): Краткое имя метрики ('auc', 'acc', 'off').
        pipe (Pipeline): Обученный пайплайн.
        train_test_dict (SplitDataset): Словарь, содержащий разделенные наборы данных.
            - X_train (pd.DataFrame): Признаки для обучающей выборки.
            - y_train (pd.Series): Целевая переменная для обучающей выборки.
            - X_test (pd.DataFrame): Признаки для тестовой выборки.
            - y_test (pd.Series): Целевая переменная для тестовой выборки.
        classes_metric_list (list[str]): Список метрик, требующих метки классов.
        y_pred (np.ndarray | None): заранее полученный предикт,
             используется если совместим с eval_metric.

    Returns:
        float | None: Значение метрики (ROC AUC или Accuracy) на тестовой выборке,
            либо None, если выбран режим 'off' или флаг не введён.
    """
    logger.info("FUNCTION compute_and_log_metrics")
    # Словарь для маппинг диспетчеризации
    eval_metrics_map = {
        "auc": evaluate_auc_score,
        "acc": evaluate_accuracy_score,
    }
    # Выбираем функцию из словаря по аргументу eval_metric
    func = eval_metrics_map.get(eval_metric)

    # В случае off или отсутствия флага
    if not func:
        logger.info("No evaluation metric selected (off mode).")
        return None
    # Получаем тестовые данные из словаря
    X_test = train_test_dict["X_test"]
    y_test = train_test_dict["y_test"]

    # Если подан y_pred нужного формата — используем его
    # Проверка размерности предикта есть в функциях модуля evaluate_metrics
    # verbose=False для отключения print() в функциях модуля evaluate_metrics
    if y_pred is not None and pred_and_metrics_compatible(
        y_pred,
        eval_metric,
        classes_metric_list,
    ):
        logger.info(f"Using provided y_pred for metric '{eval_metric}'")
        result = func(
            y_test,
            y_pred,
            verbose=False,
        )

    else:
        # Иначе делаем свежий инференс подходящего типа через pipeline
        # Если для метрики нужны метки классов
        if eval_metric in classes_metric_list:
            logger.info(f"Calculating {eval_metric.upper()}: performing predict")
            # Делаем предикт
            predictions = pipe.predict(X_test)

            # Мы ожидаем получить только массив. Если пришло что-то другое,
            # это неожиданное поведение, и мы должны немедленно упасть с ошибкой.
            if not isinstance(predictions, np.ndarray):
                raise TypeError(
                    f"Expected pipe.predict() to return np.ndarray, "
                    f"but got {type(predictions).__name__}. "
                    f"The logic for handling tuples is not implemented."
                )
            # verbose=False для отключения print() в функциях модуля evaluate_metrics
            result = func(y_test, predictions, verbose=False)
        else:
            # В остальных случаях делаем predict_proba
            logger.info(f"Calculating {eval_metric.upper()}: performing predict_proba")
            predictions = pipe.predict_proba(X_test)[:, 1]
            # verbose=False для отключения print() в функциях модуля evaluate_metrics
            result = func(y_test, predictions, verbose=False)

    return result
