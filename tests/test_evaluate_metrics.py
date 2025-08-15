import numpy as np
import pandas as pd
import pytest

# Импортируем тестируемые функции
from evaluate_metrics import (
    evaluate_auc_score,
    evaluate_accuracy_score,
    pred_and_metrics_compatible,
    compute_and_log_metrics
)

# ---------------------- ТЕСТЫ ДЛЯ evaluate_auc_score ----------------------

def test_evaluate_auc_score_binary_labels_and_probs(capsys):
    """
    Проверяем, что evaluate_auc_score:
    1. Корректно считает AUC для идеально спрогнозированных вероятностей.
    2. Возвращает значение в диапазоне [0, 1].
    3. Печатает значение в stdout при verbose=True.
    """
    # Истинные метки классов
    y_true = np.array([0, 0, 1, 1])
    # Предсказанные вероятности для класса 1
    y_score = np.array([0.1, 0.4, 0.6, 0.9])

    auc = evaluate_auc_score(
        y_true,
        y_score,
        verbose=True
    )

    # np.isclose(a, b) возвращает True, если a и b равны с учётом небольшой погрешности
    # Проверяем, что вычисленный AUC очень близок к 1.0 (идеальный прогноз)
    assert np.isclose(auc, 1.0)

    # Проверяем, что auc находится в допустимом диапазоне от 0 до 1 включительно
    assert 0.0 <= auc <= 1.0

    # capsys.readouterr() перехватывает print() в standard output
    out, _ = capsys.readouterr()
    # Проверяем что в тексте есть подстрока "AUC on test set"
    assert "AUC on test set" in out


def test_evaluate_auc_score_with_2d_array():
    """
    Проверяем, что если на вход подан двумерный массив вероятностей shape=(n, 2),
    используется второй столбец (вероятности класса 1).
    """
    y_true = np.array([0, 1])
    y_score_2d = np.array([[0.8, 0.2], [0.1, 0.9]])

    auc = evaluate_auc_score(y_true, y_score_2d, verbose=False)
    # np.isclose(a, b) возвращает True, если a и b равны с учётом небольшой погрешности
    assert np.isclose(auc, 1.0)


# ---------------------- ТЕСТЫ ДЛЯ evaluate_accuracy_score ----------------------

def test_evaluate_accuracy_score_perfect(capsys):
    """
    Проверяем, что evaluate_accuracy_score:
    1. Считает точность равной 1.0, если все прогнозы правильные.
    2. Напечатает строку с Accuracy при verbose=True.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])

    acc = evaluate_accuracy_score(y_true, y_pred, verbose=True)

    # Здесь проверяем, что accuracy == 1.0 (все ответы совпали)
    assert acc == 1.0


    # capsys.readouterr() перехватывает print() в standard output
    out, _ = capsys.readouterr()
    # проверяем наличие подстроки в выводе
    assert "Accuracy on test set" in out


def test_evaluate_accuracy_score_partial():
    """
    Проверяем, что accuracy считается верно при частично правильных прогнозах.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 0])  # 3 из 4 правильные

    acc = evaluate_accuracy_score(y_true, y_pred, verbose=False)

    # np.isclose(a, b) возвращает True, если a и b равны с учётом небольшой погрешности
    assert np.isclose(acc, 0.75)


# ---------------------- ТЕСТЫ ДЛЯ pred_and_metrics_compatible ----------------------
# @pytest.mark.parametrize — это декоратор Pytest, который позволяет
# запустить один и тот же тест с разными входными данными.
# "array,dtype,ndim,expected" — это список имён параметров тестовой функции.
# под этим декоратором тестовая функция один раз будет запущена для каждой строки списка.
@pytest.mark.parametrize("array,dtype,ndim,expected", [
    (np.array([0, 1, 0], dtype=int), int, 1, True),
    (np.array([[0, 1],[1, 0]]), int, 2, False),
    (np.array([[0.1, 0.9], [0.8, 0.2]], dtype=float), float, 2, True),
    (np.array([0.1, 0.9], dtype=float), float, 1, True),
    (np.array([0, 1], dtype=int), int, 1, False),
])
def test_pred_and_metrics_compatible(array, dtype, ndim, expected):
    """
    Проверка, что функция pred_and_metrics_compatible корректно валидирует
    формат предсказаний для классовых и вероятностных метрик.
    """
    classes_metrics = ['acc']
    eval_metric = 'acc' if np.issubdtype(array.dtype, np.integer) and array.ndim == 1 else 'auc'

    result = pred_and_metrics_compatible(
        array,
        eval_metric,
        classes_metrics
    )

    # Обычный assert равенства: проверяем, что возвращаемое функцией булево совпадает с ожидаемым
    assert result == expected


# ---------------------- ТЕСТЫ ДЛЯ compute_and_log_metrics ----------------------

def test_compute_and_log_metrics_with_y_pred_class_metric():
    """
    Если передан y_pred для accuracy, pipe не вызывается.
    """
    train_test_dict = {
        'X_test': pd.DataFrame({'x1': [1, 2, 3]}),
        'y_test': np.array([0, 1, 0])
    }
    classes_metric_list = ['acc']
    y_pred = np.array([0, 1, 0])

    result = compute_and_log_metrics(
        eval_metric='acc',
        pipe=None,
        train_test_dict=train_test_dict,
        classes_metric_list=classes_metric_list,
        y_pred=y_pred
    )

    # np.isclose — проверяем, что возвращённая accuracy близка к 1.0
    assert np.isclose(result, 1.0)


def test_compute_and_log_metrics_without_y_pred_class_metric():
    """
    Если y_pred не передан — вызывается pipe.predict.
    """

    # DummyPipe это минимальная версия pipeline
    class DummyPipe:
        def predict(self, X):
            return np.array([0, 1, 0])

    train_test_dict = {
        'X_test': pd.DataFrame({'x1': [1, 2, 3]}),
        'y_test': np.array([0, 1, 0])
    }

    result = compute_and_log_metrics(
        eval_metric='acc',
        pipe=DummyPipe(),
        train_test_dict=train_test_dict,
        classes_metric_list=['acc']
    )
    # проверяем, что точность близка к 1.0
    assert np.isclose(result, 1.0)


def test_compute_and_log_metrics_without_y_pred_prob_metric():
    """
    Если y_pred не передан, а метрика вероятностная — используется pipe.predict_proba.
    """
    class DummyPipe:
        def predict_proba(self, X):
            return np.array([[0.8, 0.2],
                             [0.1, 0.9],
                             [0.6, 0.4]])

    train_test_dict = {
        'X_test': pd.DataFrame({'x1': [1, 2, 3]}),
        'y_test': np.array([0, 1, 0])
    }

    result = compute_and_log_metrics(
        eval_metric='auc',
        pipe=DummyPipe(),
        train_test_dict=train_test_dict,
        classes_metric_list=['acc']
    )

    # Проверяем, что result лежит в нормальном диапазоне для метрик (0..1)
    assert 0.0 <= result <= 1.0
    # Проверяем, что AUC получился практически идеальным (1.0)
    assert np.isclose(result, 1.0)


def test_compute_and_log_metrics_off_mode():
    """
    В режиме 'off' функция возвращает None.
    """
    res = compute_and_log_metrics(
        eval_metric='off',
        pipe=None,
        train_test_dict={'X_test': None, 'y_test': None},
        classes_metric_list=['acc']
    )

    # Проверяем, что функция вернула None (ничего не считает)
    assert res is None
