import logging
from typing import Dict

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from decorators import memory_monitor

"""
Функции предобработки датасета.
Преобразование типов признаков в числовые.
Преобразование типов признаков к целочисленному виду разных форматов
согласно заданному словарю для оптимизации RAM.
Удаление дубликатов.
Заполнение пропусков медианами.
Из-за большого размера датасета вычисление медиан признаков занимает
большой объём памяти, что приводит к падению процесса(out-of-memory error). 
Поэтому применим кастомный imputer и будем считать медианы на 10% процентах датасета.
Оценки медианы будут приближены к реальным медианам, но не совпадать с ними.
Такое решение это компромисс между точностью и производительностью.
Например для признака id погрешность между реальной медианой и оценочной
составила около 0.05%.
"""

"""
Пайплайн препроцессинга данных (рекомендуемый порядок):
    1. Преобразование всех колонок к числовому типу (to_numeric).
    2. Импутация пропусков медианой (imputer).
    3. Преобразование к целочисленному типу (to_int).
    4. Удаление дубликатов (drop_duplicates).

Используется так:
    preprocessing_pipe = Pipeline([
        ('to_numeric', FunctionTransformer(convert_all_to_numeric_preprocessing)),
        ('imputer', imputer),
        ('to_int', FunctionTransformer(cast_columns_by_map_preprocessing)),
        ('drop_duplicates', FunctionTransformer(drop_duplicates_preprocessing)),
    ])
"""

"""
Создаём локальный логгер для этого модуля
Он наследует настройки от root logger
файла (pipeline.py)
"""
logger = logging.getLogger(__name__)

@memory_monitor
def convert_all_to_numeric_preprocessing(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Преобразует типы всех колоноки в числовые
    с заменой ошибок на NaN (errors='coerce').

    Args:
        df : Исходный DataFrame, содержащий колонки 'id' и 'rn'.

    Returns:
        pandas.DataFrame : Копия исходного DataFrame
        где все колонки приведены к числовому типу.
    """

    # errors='coerce' при невозможности преобразования заменит на NaN.
    df = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))

    return  df

@memory_monitor
def cast_columns_by_map_preprocessing(
        df: pd.DataFrame,
        cast_type_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Приводит типы указанных колонок DataFrame к типам, заданным в словаре cast_type_map.

    Args:
        df : Исходный DataFrame.
        cast_type_map : dict  Словарь соответствия {имя_колонки(str): тип(str)}.
    Returns:
        pd.DataFrame : DataFrame, где указанные колонки приведены к нужному типу.
    """

    # Согласно логике preprocessing_pipe в датасете не должно остаться NaN,
    # но всё же введём проверку на всякий случай.
    if df.isnull().any().any():
        raise ValueError(
            "Found NaN values in DataFrame. All missing values must be imputed before converting."
        )
    for col, dtype in cast_type_map.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype, copy=False)

    return df

@memory_monitor
def drop_duplicates_preprocessing(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Удаляет дубликаты строк из DataFrame.

    Args:
        df : Исходный DataFrame.

    Returns:
        pd.DataFrame : DataFrame без дублирующихся строк.
            Если дубликаты найдены и удалены — новый объект.
            Если дубликаты не найдены — возвращается исходный DataFrame без изменений.
    """

    # Подсчитываем количество дублирующихся строк
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        # Если дубликаты есть — выводим предупреждение и удаляем их
        logger.info(f"{n_dupes} duplicate records found! Removing duplicates.")

        df = df.drop_duplicates(ignore_index=True)

        return  df

    else:
        # Если дубликатов нет — уведомляем и возвращаем исходный DataFrame
        logger.info("No duplicates found, no cleanup operation required.")

        return df


class SampleMedianImputer(BaseEstimator, TransformerMixin):
    """
        Класс для imputation пропусков медианами.

        Этот трансформер наследует от BaseEstimator и TransformerMixin
        для интеграции в sklearn pipelines.
        Медианы вычисляются на подвыборке (sample_frac) для ускорения на больших датасетах.

        Args:
            sample_frac (float): Доля выборки для вычисления медиан (default 0.1, т.е. 10%).

        Attributes:
            medians_ (pd.Series): Вычисленные медианы по колонкам (сохраняются после fit).
        """
    def __init__(self, sample_frac=0.1, random_state=None):
        # Доля выборки для вычисления медиан
        self.sample_frac = sample_frac
        # Атрибут для хранения медиан
        self.medians_ = None
        self.random_state = random_state

    def fit(self, X, y=None):
        # Создаём подвыборку датасета, random_state для воспроизводимости
        sample = X.sample(frac=self.sample_frac, random_state=self.random_state)
        # Вычисляем и сохраняем медианы
        self.medians_ = sample.median()
        return self

    def transform(self, X):
        # Заполняем пропуски медианами
        return X.fillna(self.medians_)



# Добавим защитный блок main для тестов
if __name__ == "__main__":
    # Настройка логгера только для standalone запуска
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s')

    pass