import logging
import glob

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score

from config import (
    TARGET_PATH,
    TRAIN_SIZE,
    SEED_SPLIT_DATASET,
    STRATIFY_COL,
    PROBA_TEST_PREDICT_PATTERN,
    CLASSES_TEST_PREDICT_PATTERN
)

from data_utils import split_target_only

"""
Создаём локальный логгер для этого модуля
Он наследует настройки от root logger
импортирующего файла (pipeline.py)
"""
logger = logging.getLogger(__name__)


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
        eval_metric: str,
        classes_metric_list: List[str]
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
        classes_metric_list (List[str]): Список метрик, требующих метки классов.

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
    classes_metric_list: List[str],
    y_pred: Optional[np.ndarray] = None
) -> Optional[float]:
    """
    Вычисляет и логирует выбранную метрику качества (AUC или Accuracy) на тестовой выборке.

    Args:
        eval_metric (str): Краткое имя метрики ('auc', 'acc', 'off').
        pipe (Any): Обученный пайплайн.
        train_test_dict (dict): Словарь с тестовыми данными, должен содержать ключи
            'X_test' (pd.DataFrame) и 'y_test' (pd.Series или 1D np.array).
        classes_metric_list (List[str]): Список метрик, требующих метки классов.
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
    if y_pred is not None and pred_and_metrics_compatible(
            y_pred,
            eval_metric,
            classes_metric_list
    ):
        logger.info(f"Using provided y_pred for metric '{eval_metric}'")
        result = func(y_test, y_pred, verbose=False)

    else:
        # Иначе делаем свежий инференс подходящего типа через pipeline
        # Если для метрики нужны метки классов
        if eval_metric in classes_metric_list:
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
    """
    В блоке main можно посчитать метрики ROC AUC и Accuracy
    на предиктах тестового датасета.
    Из папки predictions автоматически выберутся файлы 
    типа predict__raw__2025-08-05-17-37.csv / proba__raw__2025-08-05-17-28.csv,
    созданые последними.
    """
    # Включаем вывод prints
    verbose = True

    # Получаем словарь с разделенным на train/test таргетом
    y_dict = split_target_only(
        path_to_target = TARGET_PATH,
        train_size = TRAIN_SIZE,
        random_state = SEED_SPLIT_DATASET,
        stratify_col = STRATIFY_COL,
        verbose = True
    )

    # Определяем истинные метки классов
    y_true = y_dict['y_test']

    # Находим по маске файл с предиктами вероятностей на тестовом наборе,
    # если файл еще не создан появится предупреждение
    proba_files = glob.glob(PROBA_TEST_PREDICT_PATTERN)
    if not proba_files:
        raise FileNotFoundError(
            f'No proba prediction files found for mask: {PROBA_TEST_PREDICT_PATTERN}'
        )
    # Выбираем первый файл
    proba_test_predict = proba_files[0]

    # Загружаем предикты вероятностей классов
    proba_df = pd.read_csv(proba_test_predict)
    probabilities = proba_df['proba_class_1'].values

    if verbose:
        print(
            f'Loaded predicted probabilities from {proba_test_predict}\n'
            f' shape: {probabilities.shape}'
        )

    # Находим по маске файл с предиктами вероятностей на тестовом наборе,
    # если файл еще не создан появится предупреждение
    classes_files = glob.glob(CLASSES_TEST_PREDICT_PATTERN)
    if not classes_files:
        raise FileNotFoundError(
            f'No proba prediction files found for mask: {CLASSES_TEST_PREDICT_PATTERN}'
        )
    # Выбираем первый файл
    classes_test_predict = classes_files[0]

    # Загружаем предикты классов
    classes_df = pd.read_csv(classes_test_predict)
    classes = classes_df['prediction'].values

    if verbose:
        print(
            f'Loaded predicted classes from {classes_test_predict}\n'
            f' shape: {classes.shape}'
        )
    # Проверяем совпадение длинн предикта и таргета -
    # выбрасываем предупреждение если нет.
    assert len(y_true) == len(classes), "Длины y_true и classes не совпадают!"

    # Проверяем совпадение длинн предикта и таргета -
    # выбрасываем предупреждение если нет.
    assert len(y_true) == len(probabilities), "Длины y_true и y_score не совпадают!"

    # Вызываем функцию оценки AUC
    evaluate_auc_score(
            y_true,
            probabilities,
    )

    # Вызываем функцию оценки accuracy
    evaluate_accuracy_score(
        y_true,
        classes,
    )