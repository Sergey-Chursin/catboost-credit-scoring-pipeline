import logging
from typing import Any, cast

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.decorators import memory_monitor_function

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
Настройки (уровень логирования, формат сообщений) наследуются от root logger, 
который обычно конфигурируется в главном файле проекта (pipeline.py).
"""
logger = logging.getLogger(__name__)


@memory_monitor_function
def convert_all_to_numeric_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует типы всех колоноки в числовые
    с заменой ошибок на NaN (errors='coerce').

    Args:
        df (pd.DataFrame): Исходный DataFrame, содержащий колонки 'id' и 'rn'.

    Returns:
        pd.DataFrame : DataFrame
        где все колонки приведены к числовому типу.
    """

    # errors='coerce' при невозможности преобразования заменит на NaN.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@memory_monitor_function
def cast_columns_by_map_preprocessing(
    df: pd.DataFrame,
    cast_type_map: dict[str, str],
) -> pd.DataFrame:
    """
    Приводит типы указанных колонок DataFrame к типам, заданным в словаре cast_type_map.

    Args:
        df (pd.DataFrame): Исходный DataFrame.
        cast_type_map (dict[str, str]): Словарь соответствия {имя_колонки(str): тип(str)}.
    Returns:
        pd.DataFrame : DataFrame, где указанные колонки приведены к нужному типу.
    """

    # Согласно логике preprocessing_pipe в датасете не должно остаться NaN,
    # но для безопасности введём проверку.
    if df.isnull().any().any():
        raise ValueError(
            "Found NaN values in DataFrame. All missing values must be imputed before converting."
        )
    for col, dtype in cast_type_map.items():
        if col in df.columns:
            df[col] = df[col].astype(cast(Any, dtype), copy=False)
    #  cast(Any, dtype) - явно указывает mypy что мы уверены в типе к которому приводим колонку
    return df


@memory_monitor_function
def drop_duplicates_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет дубликаты строк из DataFrame.

    Args:
        df (pd.DataFrame): Исходный DataFrame.

    Returns:
        pd.DataFrame : DataFrame без дублирующихся строк.
    """

    # Подсчитываем количество дублирующихся строк
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        # Если дубликаты есть — выводим предупреждение и удаляем их
        logger.info(f"{n_dupes} duplicate records found! Removing duplicates.")

        df = df.drop_duplicates(ignore_index=True)
        return df

    else:
        # Если дубликатов нет — уведомляем и возвращаем исходный DataFrame
        logger.info("No duplicates found, no cleanup operation required.")
        return df


class SampleMedianImputer(BaseEstimator, TransformerMixin):
    """
    Импутер для заполнения пропусков в признаках датасета медианами
    вычисленными на подвыборке датасета для ускорения вычислений на больших датасетах.
    Оценки медианы будут приближены к реальным медианам, но не совпадать с ними.
    Такое решение это компромисс между точностью и производительностью.

    Attributes:
        sample_frac (float): Доля выборки для вычисления медиан.
        random_state (int): Зерно рандома для воспроизводимости результатов.
        medians_ (pd.Series | None): Series с медианами столбцов, вычисляется после fit.
    """

    def __init__(
        self,
        sample_frac: float = 0.1,
        random_state: int | None = None,
    ):
        """
        Args:
            sample_frac (float): Доля выборки для вычисления медиан.
                По умолчанию 0.1 (10%).
            random_state (int | None): Зерно рандома для воспроизводимости результатов.
                По умолчанию None.
        """
        self.sample_frac: float = sample_frac
        self.random_state: int = random_state if random_state is not None else 0
        self.medians_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: None = None) -> "SampleMedianImputer":
        """
        Создаёт атрибут medians_ - pd.Series содержащий медианы признаков подвыборки датасета.
        Args:
            X (pd.DataFrame): Тренировочный датафрейм.
            y (None): Добавлен для совместимости с родительским классом BaseEstimator sklearn.
        Returns:
            self : SampleMedianImputer - обученный импутер.
        """
        # Создаём подвыборку датасета
        sample = X.sample(frac=self.sample_frac, random_state=self.random_state)
        # Вычисляем и сохраняем медианы
        self.medians_ = sample.median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Заполняет пропуски в признаках датасета медианами.
        Args:
            X (pd.DataFrame): Датафрейм для обработки.
        Returns:
            X (pd.DataFrame): Обработанный Датафрейм.
        """
        return X.fillna(self.medians_)
