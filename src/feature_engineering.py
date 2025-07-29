import logging
import numpy as np
import pandas as pd
from config import (
    PROP_FEATURES_DICT,
    MEAN_FREQ_SOURCE_LIST,
    DROP_LIST
)
"""
Создаём локальный логгер для этого модуля
Он наследует настройки от root logger
импортирующего файла (pipeline.py)
"""
logger = logging.getLogger(__name__)

# FEATURE ENGINEERING PIPELINE FUNCTIONS

def rn_max_feature_pipeline(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Добавляет в DataFrame новую колонку 'rn_max' — максимальное
    значение 'rn' для каждой группы 'id'.

    Args:
        df : Исходный DataFrame, содержащий колонки 'id' и 'rn'.

    Returns:
        pandas.DataFrame : Копия исходного DataFrame с добавленной колонкой 'rn_max'.
    """
    logger.info('FUNCTION rn_max_feature_pipeline')

    """
    Для каждой строки определяем максимальное значение 'rn' среди всех строк с тем же 'id'
    Метод transform('max') возвращает Series длины исходного DataFrame, где для каждой строки
    указано максимальное значение 'rn' в её группе 'id'.
    """
    df['rn_max'] = df.groupby('id')['rn'].transform('max')

    return df


def enc_paym_transcoding_pipeline(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Прекодирует признаки enc_paym_features к единому виду с диапазоном значений {0, 1, 2, 3}.
    Для каждого столбца enc_paym_0, enc_paym_1, ..., enc_paym_24,
    если в значениях встречается 4, происходит замена:
        1 -> 0
        2 -> 1
        3 -> 2
        4 -> 3

    Args:
        df : Исходный DataFrame с колонками 'enc_paym_0' ... 'enc_paym_24'.

    Returns:
    pandas.DataFrame : Копия DataFrame с перекодированными признаками.
    """
    logger.info('FUNCTION enc_paym_transcoding_pipeline ')

    # Список колонок для перекодировки
    columns = [f'enc_paym_{i}' for i in range(25)]

    for col in columns:
        # Проверяем, есть ли значение 4 в колонке
        if 4 in df[col].unique():
            # Заменяем значения согласно маппингу
            df.loc[:, col] = df[col].replace({1: 0, 2: 1, 3: 2, 4: 3})

    return df


def definite_value_proportion_features_pipeline(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Создаёт и добавляет в датафрейм новые частотные признаки
    на основе заданных значений исходных признаков.

    Для каждого столбца и каждого указанного значения в словаре функция создаёт новые признаки,
    отражающие долю записей с этим значением относительно общего количества
    кредитов (rn_max) для каждого id.

    Args:
        df : Исходный DataFrame, содержащий необходимые признаки и колонку 'rn_max'.

    Returns:
        pandas.DataFrame : Копия исходного DataFrame с добавленными частотными признаками.
    """
    logger.info('FUNCTION definite_value_proportion_features_pipeline')

    """
    Создадим словарь где для каждого признака перечислены значения,
    по которым считаем долю.
    """
    features_dictionary = PROP_FEATURES_DICT

    # Итерируем по ключам
    for col in features_dictionary.keys():
        logger.info('Original feature %s', col)
        logger.info('New features')

        # Итерируем по значениям
        for value in features_dictionary[col]:
            new_column = f'{col}_prop_{value}'
            logger.info(new_column)

            """
            Создаём булевую маску: True, если значение в col равно value,
            иначе False.
            """
            mask = (df[col] == value)
            """
            Для каждой строки вычисляем количество совпадений value 
            по id (transform('sum')) и делим на общее количество кредитов 
            по id (rn_max), чтобы получить долю.
            """
            df[new_column] = mask.groupby(df['id']).transform('sum') / df['rn_max']

    return df


def from_is_zero_prop_1_create_sum_prop_1_feature_pipeline(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Вычисляет среднее значение признаков is_zero_*_prop_1 по строкам и добавляет
    новый признак 'is_zero_sum_prop_1' в DataFrame.

    Args:
        df :  Исходный DataFrame с признаками is_zero_*_prop_1.

    Returns:
        pandas.DataFrame : Копия DataFrame с добавленным признаком 'is_zero_sum_prop_1'.
    """
    logger.info('FUNCTION from_is_zero_prop_1_create_sum_prop_1_feature_pipeline')

    columns = [
        'is_zero_loans5_prop_1',
        'is_zero_loans530_prop_1',
        'is_zero_loans3060_prop_1',
        'is_zero_loans6090_prop_1',
        'is_zero_loans90_prop_1'
    ]

    df['is_zero_sum_prop_1'] = df[columns].sum(axis=1) / 5

    return df


def mean_value_frequency_feature_pipeline(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Cоздаёт новые агрегированные признаки,
    отражающий среднюю частоту (относительную встречаемость) значений
    заданных столбцов columns_list датафрейма для каждого уникального id.
    Результат добавляется в  датафрейм
    с нормировкой на количество записей (rn_max) для каждого id.

    Args:
        df :  Исходный DataFrame с признаками из columns_list.

    Returns:
        pandas.DataFrame :  Копия DataFrame с добавленным новым столбцом {column}_mean_freq,
        содержащим нормированное агрегированное значение средней
        частоты значений column для каждого id.
    """
    logger.info('FUNCTION mean_value_frequency_feature_pipeline')

    # Список столбцов, для которых считаем среднюю частоту значений
    columns_list = MEAN_FREQ_SOURCE_LIST

    logger.info('New features')

    for col in columns_list:
        new_column = f'{col}_mean_freq'
        logger.info(new_column)

        # Вычисляем относительную частоту каждого уникального значения в столбце
        bin_freq = df[col].value_counts(normalize=True).to_dict()

        # Создаём Series с частотами значений для каждой строки
        freq_series = df[col].map(bin_freq)
        """
        Для каждой строки считаем сумму частот значений (freq_series) по группе 'id'.
        Делим эту сумму на общее количество записей по id (rn_max),
        чтобы получить нормированную среднюю частоту встречаемости значений
        признака для данного id.
        Результат сохраняем в новый столбец new_column.
        """
        df[new_column] = freq_series.groupby(df['id']).transform('sum') / df['rn_max']

    return df


def enc_paym_norm_group_sum_diff_pipeline(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Генерирует признаки разницы между средними количествами различных статусов платежей
    по кредитам за разные временные промежутки.

    Основная цель функции — создать итоговые признаки:
        - 'enc_paym_avg_0_1_this_year_diff'
        - 'enc_paym_avg_1_2_all_diff'
        - 'enc_paym_avg_0_years_diff'

    Для их расчёта временно создаются промежуточные агрегированные признаки среднего
    количества статусов платежей по id и периоду
    (например, 'enc_paym_avg_0_this_year'),
    которые впоследствии удаляются из итогового датасета.

    Args:
        df :  Исходный DataFrame с признаками из columns_list.

    Returns:
        pandas.DataFrame :  Копия DataFrame с добавленными итоговыми признаками
        разницы между средними количествами статусов платежей по различным периодам.
    """

    logger.info('FUNCTION enc_paym_norm_group_sum_diff_pipeline')
    logger.info('New features')

    # Создаём временный датафрейм со столбцом id из df
    df_buff = pd.DataFrame(data=df['id'], columns=['id'])

    # Временной промежуток 'all' — все периоды
    time_span = 'all'
    columns = [f'enc_paym_{i}' for i in range(25)]

    # Для статусов платежей по кредитам 1 и 2
    for i in range(1, 3):
        new_col = f'enc_paym_avg_{i}_{time_span}'
        logger.info(new_col)

        # Считаем количество статуса i по всем столбцам за период
        df_buff[new_col] = np.sum(
            [df[col] == i for col in columns],
            axis=0
        )
        """
        Суммируем значения признака new_col по всем строкам с одинаковым id,
        затем делим на количество записей по этому id (rn_max),
        чтобы получить среднее количество появлений статуса для каждой строки.
        Cохраняем результат в новый столбец DataFrame с именем new_col.
        """
        df[new_col] = (
                df_buff[new_col].groupby(df_buff['id']).transform('sum')
                / df['rn_max']
        )

    # Временной промежуток 'this_year' — первые 12 месяцев
    time_span = 'this_year'
    columns = [f'enc_paym_{i}' for i in range(12)]

    # Для статусов платежей по кредитам 0 и 1
    for i in range(2):
        new_col = f'enc_paym_avg_{i}_{time_span}'
        logger.info(new_col)

        # Считаем количество статуса i по всем столбцам за период
        df_buff[new_col] = np.sum(
            [df[col] == i for col in columns],
            axis=0
        )
        """
        Суммируем значения признака new_col по всем строкам с одинаковым id,
        затем делим на количество записей по этому id (rn_max),
        чтобы получить среднее количество появлений статуса для каждой строки.
        Cохраняем результат в новый столбец DataFrame с именем new_col.
        """
        df[new_col] = (
                df_buff[new_col].groupby(df_buff['id']).transform('sum')
                / df['rn_max']
        )

    # Временной промежуток 'last_year' — месяцы с 12 по 24
    time_span = 'last_year'
    columns = [f'enc_paym_{i}' for i in range(12, 25)]

    """
    Статус платежей  0.
    (Оставим цикл для единообразия кода)
    """
    for i in [0]:
        new_col = f'enc_paym_avg_{i}_{time_span}'
        logger.info(new_col)

        # Считаем количество статуса i по всем столбцам за период
        df_buff[new_col] = np.sum(
            [df[old_col] == i for old_col in columns],
            axis=0
        )
        """
        Суммируем значения признака new_col по всем строкам с одинаковым id,
        затем делим на количество записей по этому id (rn_max),
        чтобы получить среднее количество появлений статуса для каждой строки.
        Cохраняем результат в новый столбец DataFrame с именем new_col.
        """
        df[new_col] = (
                df_buff[new_col].groupby(df_buff['id']).transform('sum')
                / df['rn_max']
        )

    # Создаём фичи разницы
    df['enc_paym_avg_0_1_this_year_diff'] = (
            df['enc_paym_avg_0_this_year'] -
            df['enc_paym_avg_1_this_year']
    )

    df['enc_paym_avg_1_2_all_diff'] = (
            df['enc_paym_avg_1_all'] -
            df['enc_paym_avg_2_all']
    )

    df['enc_paym_avg_0_years_diff'] = (
            df['enc_paym_avg_0_this_year'] -
            df['enc_paym_avg_0_last_year']
    )

    logger.info("""\
New difference columns
enc_paym_avg_0_1_this_year_diff
enc_paym_avg_1_2_all_diff
enc_paym_avg_0_years_diff
"""
                )

    return df


def pre_since_opened_sum_mean_repeated_pipeline(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Cоздаёт признак, отражающий пропорцию повторяющихся значений 'pre_since_opened'
    для каждого 'id'.

    Логика работы:
    - Подсчитывает количество появлений каждого значения 'pre_since_opened' для каждого 'id'.
    - Выделяет только повторяющиеся значения (где количество > 1) и вычитает 1,
      чтобы не считать первое появление.
    - Суммирует количество повторов по всем значениям 'pre_since_opened' для каждого 'id'.
    - Добавляет отсутствующие 'id' с нулевыми значениями повторов.
    - Добавляет новый признак 'pre_since_opened_repeated_prop' в df_to_update,
      нормируя сумму повторов на количество записей 'rn_max' для каждого 'id'.

    Args:
        df :  Исходный DataFrame с признаками  'pre_since_opened', 'id' и 'rn_max'.

    Returns:
        pandas.DataFrame :  Копия DataFrame с
        добавленным признаком 'pre_since_opened_repeated_prop'.
    """
    logger.info('FUNCTION pre_since_opened_sum_mean_repeated_pipeline')

    # Считаем количество каждого значения 'pre_since_opened' для каждого 'id'
    counts = df.groupby(['id', 'pre_since_opened']).size()

    """
    Оставляем только повторяющиеся значения (количество > 1), 
    вычитаем первое появление.
    """
    repeated_pre_since_opened = counts[counts > 1] - 1

    # Суммируем количество повторов по каждому 'id'
    sum_repeated = repeated_pre_since_opened.groupby('id').sum()

    # Добавляем отсутствующие 'id' с нулевыми значениями повторов
    all_sum_repeated = sum_repeated.reindex(df['id'].unique(), fill_value=0)

    # Добавляем новый столбец: для каждого 'id' записываем рассчитанную сумму повторов
    df['pre_since_opened_repeated_prop'] = df['id'].map(all_sum_repeated)

    # Нормируем сумму повторов на количество записей 'rn_max' для каждого 'id'
    df['pre_since_opened_repeated_prop'] = (
            df['pre_since_opened_repeated_prop'] / df['rn_max']
    )

    return df


def drop_columns_drop_duplicates_pipeline(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Удаляет исходные и временные признаки из DataFrame,
    а также удаляет дубликаты по столбцу 'id', оставляя только первую запись.
    После удаления дубликатов столбец 'id' также удаляется.

    Args:
        df : Исходный DataFrame.

    Returns:
        pd.DataFrame : Копия DataFrame без указанных столбцов и дубликатов по 'id'.
    """

    logger.info('FUNCTION drop_columns_drop_duplicates_pipeline')
    # Список столбцов на удаление
    columns = DROP_LIST

    df = df.drop(columns, axis=1)

    """
    Удаляем дубликаты по столбцу 'id', оставляя первую запись
    и сбрасываем индекс.
    """
    df = df.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)

    # Удаляем столбец 'id', так как он больше не нужен
    df = df.drop('id', axis=1)

    return df

# Добавим защитный блок main для тестов
if __name__ == "__main__":
    # Настройка логгера только для standalone запуска
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s')

    pass