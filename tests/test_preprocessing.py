import pandas as pd
import numpy as np
import pytest
from src.preprocessing import (
    convert_all_to_numeric_preprocessing,
    cast_columns_by_map_preprocessing,
    drop_duplicates_preprocessing,
    SampleMedianImputer
)


def test_convert_all_to_numeric_preprocessing():
    """
    Проверка, что функция корректно приводит все колонки DataFrame к числовому типу (float),
    нечисловые значения становятся NaN.
    """
    df = pd.DataFrame({
        'id': ['1', '2', '3'],
        'value': ['42.5', 'not_a_num', '10']
    })
    result = convert_all_to_numeric_preprocessing(df)
    # Все данные стали float-типами, нечисловое значение стало NaN
    # функция np.issubdtype(dtype, np.number) возвращает True,
    # если тип данных столбца — любой числовой тип (целый, дробный, np.float32, np.float64, np.int64 и т.д.)
    assert np.issubdtype(result['id'].dtype, np.number)
    # pandas при появлении NaN всегда переводит столбец в float.
    assert result['value'].dtype == float
    # Проверяем что 'not_a_num' стал NaN
    assert np.isnan(result['value'][1])


def test_cast_columns_by_map_preprocessing_success():
    """
    Проверяет, что DataFrame без NaN после вызова функции становится с типом int.
    """
    df = pd.DataFrame({
            'a': [1., 2., 3.],
            'b': [0., -3.3, 14.],
            'c': [5, 3.3, 8]
        })
    result = cast_columns_by_map_preprocessing(
        df,
        cast_type_map={
        'a': 'int32',
        'b': 'float64',
        'c': 'str'
        }
    )
    assert np.issubdtype(result['a'].dtype, np.integer)
    assert np.issubdtype(result['b'].dtype, np.floating)
    assert result['c'].dtype == object
    assert (result['a'] == [1, 2, 3]).all()
    assert (result['b'] == [0., -3.3, 14.]).all()


def test_cast_columns_by_map_preprocessing_with_nan_raises():
    """
    Проверяет, что при наличии NaN в DataFrame функция вызывает ValueError.
    """
    df_with_nan = pd.DataFrame({'a': [1., np.nan, 3.]})
    # Используем контекстный менеджер проверяющий вызов ошибки при запуске функции
    with pytest.raises(ValueError):
        cast_columns_by_map_preprocessing(df_with_nan, cast_type_map={})


def test_drop_duplicates_pipeline():
    """
    Проверяет, что функция удаляет полностью одинаковые строки.
    """
    df = pd.DataFrame({'a': [1, 1, 2], 'b': [5, 5, 6]})
    result = drop_duplicates_preprocessing(df)
    # Должны остаться только уникальные строки
    assert len(result) == 2
    assert sorted(result['a'].tolist()) == [1, 2]
    assert sorted(result['b'].tolist()) == [5, 6]


def test_SampleMedianImputer_fit_transform_reproducible():
    """
    Проверяет, что кастомный SampleMedianImputer правильно вычисляет медианы на подвыборке,
    их запоминает, и корректно заполняет пропуски.
    """
    df = pd.DataFrame({
        'a': [1, 2, np.nan, 4, 9, 3, 10, np.nan, 1, 0],
        'b': [np.nan, np.nan, 3, 4, 4, 0, 4, np.nan, 20, 1]
    })
    # Сделаем небольшую подвыборку для медиан
    imputer = SampleMedianImputer(sample_frac=0.6, random_state=42)
    imputer.fit(df)
    # Сохранили медианы (на подвыборке! — могут быть неидеальны)
    # Проверим, что медианы — числа
    assert isinstance(imputer.medians_['a'], float)
    assert isinstance(imputer.medians_['b'], float)
    # transform должен заполнить NaN медианными значениями
    result = imputer.transform(df)
    # убедимся, что все NaN заполнены
    assert not result.isnull().any().any()


def test_SampleMedianImputer_fit_small_sample():
    """
    Проверяет, что медианный заполнитель работает, даже если подвыборка — одна строка (sample_frac small).
    """
    df = pd.DataFrame({'a': [10, np.nan, 20]})
    imputer = SampleMedianImputer(sample_frac=0.34, random_state=0)
    imputer.fit(df)
    # Проверим, что transform не ломается
    result = imputer.transform(df)
    assert not result.isnull().any().any()


def test_preprocessing_pipeline_sample():
    """
    Интеграционный тест: пропуск через несколько функций пайплайна на реальных мини-данных.
    """
    df = pd.DataFrame({
        'id': ['1', '2', '3', 'not_int'],
        'x': ['1', 'NaN', '17.2', '10.5'],
        'y': ['1', '42', '-1', '1']
    })

    #  Преобразуем к числовым
    out = convert_all_to_numeric_preprocessing(df)
    #  Импутируем NaN
    imputer = SampleMedianImputer(sample_frac=0.5, random_state=0)
    imputer.fit(out)
    out = imputer.transform(out)
    # Преобразуем к int
    out = cast_columns_by_map_preprocessing(out, cast_type_map={})
    # Удаляем дубликаты
    out = drop_duplicates_preprocessing(out)

    # Проверка: нет NaN, нет дубликатов
    assert not out.isnull().any().any()
    assert len(out) == len(out.drop_duplicates())

