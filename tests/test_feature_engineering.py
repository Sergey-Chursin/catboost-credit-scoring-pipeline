import pandas as pd
import numpy as np

from feature_engineering import (
    rn_max_feature_pipeline,
    enc_paym_transcoding_pipeline,
    definite_value_proportion_features_pipeline,
    from_is_zero_prop_1_create_sum_prop_1_feature_pipeline,
    mean_value_frequency_feature_pipeline,
    drop_columns_drop_duplicates_pipeline,
    pre_since_opened_sum_mean_repeated_pipeline
)

def test_rn_max_feature_pipeline_basic():
    """
    Проверяет, что функция корректно добавляет колонку rn_max с максимальным rn по id.
    """
    # Пример данных: два id, по два кредита у первого id, один у второго
    df = pd.DataFrame({
        "id": [1, 1, 2],
        "rn": [10, 5, 3]
    })
    result = rn_max_feature_pipeline(df.copy())
    # Проверяем, что добавилась колонка rn_max
    assert 'rn_max' in result.columns
    # Для id=1 должно быть max=10, для id=2 — max=3
    assert all(np.isclose(result[result["id"] == 1]['rn_max'], 10))
    assert all(np.isclose(result[result["id"] == 2]['rn_max'], 3))


def test_enc_paym_transcoding_pipeline_recode():
    """
    Проверка перекодировки: 1->0, 2->1, 3->2, 4->3 в enc_paym_* колонках.
    """
    df = pd.DataFrame({
        "enc_paym_0": [1, 3, 4],
        "enc_paym_1": [2, 2, 3]
    })
    result = enc_paym_transcoding_pipeline(df.copy())
    # Проверяем, что значения заменились правильно
    assert result["enc_paym_0"].tolist() == [0, 2, 3]
    assert result["enc_paym_1"].tolist() == [1, 1, 2]

def test_definite_value_proportion_features_pipeline():
    """
    Проверяет, что функция для каждого значения из словаря создает новую колонку,
    в которой — доля его появления по id, деленная на rn_max.
    """
    # у id=1 две строки — одна с feature=2, другая с feature=1.
    # У id=2 одна строка — feature=2.
    df = pd.DataFrame({
        "id": [1, 1, 2],
        "feature": [2, 1, 2],
        "rn_max": [2, 2, 1]
    })
    # Хочем считать долю встретившихся "2" и "1" в колонке feature
    features_dict = {"feature": [2, 1]}
    result = definite_value_proportion_features_pipeline(df.copy(), features_dict)
    # Доля "2" для id=1: 1/2, для id=2: 1/1
    # Доля "1" для id=1: 1/2, для id=2: 0/1
    assert "feature_prop_2" in result.columns
    assert "feature_prop_1" in result.columns
    assert list(result["feature_prop_2"].loc[result["id"] == 1]) == [0.5, 0.5]
    assert result["feature_prop_2"].loc[result["id"] == 2].iloc[0] == 1.0
    assert list(result["feature_prop_1"].loc[result["id"] == 1]) == [0.5, 0.5]
    assert (result["feature_prop_1"].loc[result["id"] == 2] == 0.0).all()


def test_from_is_zero_prop_1_create_sum_prop_1_feature_pipeline():
    """
    Проверяет создание агрегирующего признака (среднее от пяти колонок).
    """
    df = pd.DataFrame({
        'is_zero_loans5_prop_1': [1, 0],
        'is_zero_loans530_prop_1': [0, 1],
        'is_zero_loans3060_prop_1': [1, 0],
        'is_zero_loans6090_prop_1': [1, 1],
        'is_zero_loans90_prop_1': [1, 0]
    })
    result = from_is_zero_prop_1_create_sum_prop_1_feature_pipeline(df.copy())
    # Для первой строки: (1+0+1+1+1)/5 = 0.8, для второй: (0+1+0+1+0)/5 = 0.4
    assert np.isclose(result['is_zero_sum_prop_1'].iloc[0], 0.8)
    assert np.isclose(result['is_zero_sum_prop_1'].iloc[1], 0.4)

def test_mean_value_frequency_feature_pipeline():
    """
    Проверяет, что считается средняя частота (mean freq) значений колонки по id.
    """
    df = pd.DataFrame({
        "id": [1, 1, 2, 2],
        "some_col": [2, 2, 1, 1],
        "rn_max": [2, 2, 1, 1]
    })
    result = mean_value_frequency_feature_pipeline(df.copy(), columns_list=["some_col"])
    # Значения "2" у id=1 встречаются всегда → freq=1.0
    # Значения "1" у id=2 встречаются всегда → freq=1.0
    assert all(result.query("id == 1")["some_col_mean_freq"] == 0.5)
    assert all(result.query("id == 2")["some_col_mean_freq"] == 1)


def test_pre_since_opened_sum_mean_repeated_pipeline():
    """
    Проверка вычисления пропорции повторов pre_since_opened по id, с учетом деления на rn_max.
    """
    df = pd.DataFrame({
        'id': [1, 1, 1, 2, 2],
        'pre_since_opened': [5, 5, 6, 6, 7],
        'rn_max': [3, 3, 3, 2, 2]
    })
    # id=1: 5 дважды (повтор 1 раз), 6 один раз → повторов 1
    # prop = 1/3 для всех строк с id=1
    # id=2: 6 и 7 — по одному разу (повторов нет), пропорция — 0
    result = pre_since_opened_sum_mean_repeated_pipeline(df.copy())
    assert all(np.isclose(result.query("id==1")["pre_since_opened_repeated_prop"], 1/3))
    assert all(np.isclose(result.query("id==2")["pre_since_opened_repeated_prop"], 0.0))

def test_drop_columns_drop_duplicates_pipeline():
    """
    Проверяет, что функция удаляет заданные колонки, дубликаты по id и сам id.
    """
    df = pd.DataFrame({
        "id": [1, 1, 2],
        "a": [10, 20, 30],
        "b": [100, 200, 300]
    })
    result = drop_columns_drop_duplicates_pipeline(df.copy(), columns_list=["a"])
    # После удаления колонки 'a', дубликатов по id и самой 'id', остаётся только одна колонка 'b'
    assert 'a' not in result.columns
    assert 'id' not in result.columns
    assert list(result.columns) == ['b']
    # Остаётся только 2 строки (по числу уникальных id)
    assert result.shape[0] == 2