import logging

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

"""
Функции предобработки датасета.
Преобразование типов признаков в числовые.
Преобразование типов признаков к целочисленному виду.
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
        ('to_numeric', FunctionTransformer(convert_all_to_numeric_pipeline)),
        ('imputer', imputer),
        ('to_int', FunctionTransformer(convert_all_to_int_pipeline)),
        ('drop_duplicates', FunctionTransformer(drop_duplicates_pipeline)),
    ])
"""

"""
Создаём локальный логгер для этого модуля
Он наследует настройки от root logger
импортирующего файла (pipeline.py)
"""
logger = logging.getLogger(__name__)

def convert_all_to_numeric_pipeline(
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
    logger.info('FUNCTION convert_all_to_numeric_pipeline')

    # Копируем датасет чтобы не изменять оригинал.
    df = df.copy()
    # errors='coerce' при невозможности преобразования заменит на NaN.
    return df.apply(lambda col: pd.to_numeric(col, errors='coerce'))


def convert_all_to_int_pipeline(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Преобразует все колонки DataFrame к целочисленному типу.

    Args:
        df : Исходный DataFrame с числовыми значениями.

    Returns:
        pd.DataFrame : DataFrame, где все колонки приведены к типу int.
    """
    logger.info('FUNCTION convert_all_to_int_pipeline')


    # Согласно логике preprocessing_pipe в датасете не должно остаться NaN,
    # но всё же введём проверку на всякий случай.

    if df.isnull().any().any():
        raise ValueError(
            "Found NaN values in DataFrame. All missing values must be imputed before converting to int."
        )

    return df.astype(int)

def drop_duplicates_pipeline(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Удаляет дубликаты строк из DataFrame.

    Args:
        df : Исходный DataFrame.

    Returns:
        pd.DataFrame : DataFrame без дублирующихся строк.
    """

    logger.info('FUNCTION drop_duplicates_pipeline')

    return df.drop_duplicates()


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