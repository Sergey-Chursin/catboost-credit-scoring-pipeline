import gc
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.decorators import memory_monitor_function
from src.memory_utils import cgroup_memory_statistic, heap_trim, rss_process_statistic

"""
Создаём локальный логгер для этого модуля
Он наследует настройки от root logger
 файла (pipeline.py)
"""
logger = logging.getLogger(__name__)

# FEATURE ENGINEERING PIPELINE FUNCTIONS


@memory_monitor_function
def rn_max_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет в DataFrame новую колонку 'rn_max' — максимальное
    значение 'rn' для каждой группы 'id'.
    Удаляет исходную колонку 'rn'.

    Args:
        df : Исходный DataFrame, содержащий колонки 'id' и 'rn'.

    Returns:
        pandas.DataFrame : Копия исходного DataFrame с добавленной колонкой 'rn_max'.
    """

    # Для каждой строки определяем максимальное значение 'rn' среди всех строк с тем же 'id'
    # Метод transform('max') возвращает Series длины исходного DataFrame, где для каждой строки
    # указано максимальное значение 'rn' в её группе 'id'.

    df["rn_max"] = df.groupby("id")["rn"].transform("max")

    # Удаляем уже не нужный столбец для экономии памяти
    df = df.drop("rn", axis=1)

    return df


@memory_monitor_function
def enc_paym_transcoding_pipeline(df: pd.DataFrame) -> pd.DataFrame:
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

    # Список колонок для перекодировки
    columns = [col for col in df.columns if col.startswith("enc_paym_")]

    # Заменяем значения в любом случае, а не только если есть 4
    for col in columns:
        df.loc[:, col] = df[col].replace({1: 0, 2: 1, 3: 2, 4: 3})

    return df


@memory_monitor_function
def enc_paym_norm_group_sum_diff_pipeline(
    df: pd.DataFrame, drop_list: List[str]
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
        drop_list: List[str]: Список уже не нужных признаков,
            для удаления из датафрейма

    Returns:
        pandas.DataFrame :  Копия DataFrame с добавленными итоговыми признаками
        разницы между средними количествами статусов платежей по различным периодам.

    ВАЖНО: функция требует два аргумента на входе, что несовместимо с работой sklearn Pipeline
    (который ожидает функцию только с одним аргументом — DataFrame).
     Поэтому при добавлении этой функции в пайплайн её необходимо оборачивать с помощью partial,
      чтобы зафиксировать дополнительные параметры заранее.
    """

    logger.info("NEW temporary features")

    # Создаём временный датафрейм со столбцом id из df
    df_buff = pd.DataFrame(data=df["id"], columns=["id"])

    # Временной промежуток 'all' — все периоды
    time_span = "all"
    columns = [f"enc_paym_{i}" for i in range(25)]

    # Для статусов платежей по кредитам 1 и 2
    for i in range(1, 3):
        new_col = f"enc_paym_avg_{i}_{time_span}"
        logger.info(new_col)

        # Считаем количество статуса i по всем столбцам за период
        df_buff[new_col] = np.sum([df[col] == i for col in columns], axis=0)
        """
        Суммируем значения признака new_col по всем строкам с одинаковым id,
        затем делим на количество записей по этому id (rn_max),
        чтобы получить среднее количество появлений статуса для каждой строки.
        Cохраняем результат в новый столбец DataFrame с именем new_col.
        """
        df[new_col] = (
            df_buff[new_col].groupby(df_buff["id"]).transform("sum") / df["rn_max"]
        )

    # Временной промежуток 'this_year' — первые 12 месяцев
    time_span = "this_year"
    columns = [f"enc_paym_{i}" for i in range(12)]

    # Для статусов платежей по кредитам 0 и 1
    for i in range(2):
        new_col = f"enc_paym_avg_{i}_{time_span}"
        logger.info(new_col)

        # Считаем количество статуса i по всем столбцам за период
        df_buff[new_col] = np.sum([df[col] == i for col in columns], axis=0)
        """
        Суммируем значения признака new_col по всем строкам с одинаковым id,
        затем делим на количество записей по этому id (rn_max),
        чтобы получить среднее количество появлений статуса для каждой строки.
        Cохраняем результат в новый столбец DataFrame с именем new_col.
        """
        df[new_col] = (
            df_buff[new_col].groupby(df_buff["id"]).transform("sum") / df["rn_max"]
        )

    # Временной промежуток 'last_year' — месяцы с 12 по 24
    time_span = "last_year"
    columns = [f"enc_paym_{i}" for i in range(12, 25)]

    """
    Статус платежей  0.
    (Оставим цикл для единообразия кода)
    """
    for i in [0]:
        new_col = f"enc_paym_avg_{i}_{time_span}"
        logger.info(new_col)

        # Считаем количество статуса i по всем столбцам за период
        df_buff[new_col] = np.sum([df[old_col] == i for old_col in columns], axis=0)
        """
        Суммируем значения признака new_col по всем строкам с одинаковым id,
        затем делим на количество записей по этому id (rn_max),
        чтобы получить среднее количество появлений статуса для каждой строки.
        Cохраняем результат в новый столбец DataFrame с именем new_col.
        """
        df[new_col] = (
            df_buff[new_col].groupby(df_buff["id"]).transform("sum") / df["rn_max"]
        )

    # Удаляем временный df_buff
    del df_buff
    gc.collect()

    # Создаём фичи разницы
    df["enc_paym_avg_0_1_this_year_diff"] = (
        df["enc_paym_avg_0_this_year"] - df["enc_paym_avg_1_this_year"]
    )

    df["enc_paym_avg_1_2_all_diff"] = (
        df["enc_paym_avg_1_all"] - df["enc_paym_avg_2_all"]
    )

    df["enc_paym_avg_0_years_diff"] = (
        df["enc_paym_avg_0_this_year"] - df["enc_paym_avg_0_last_year"]
    )

    logger.info("""\
NEW difference columns
enc_paym_avg_0_1_this_year_diff
enc_paym_avg_1_2_all_diff
enc_paym_avg_0_years_diff
""")

    # Удаляем уже не нужные колонки
    df = df.drop(drop_list, axis=1)
    logger.info(f"DataFrame shape after drop(): {df.shape}")

    return df


@memory_monitor_function
def mean_value_frequency_feature_pipeline(
    df: pd.DataFrame, columns_list: List[str], drop_list: List[str] = None
) -> pd.DataFrame:
    """
    Cоздаёт новые агрегированные признаки,
    отражающий среднюю частоту (относительную встречаемость) значений
    заданных столбцов columns_list датафрейма для каждого уникального id.
    Результат добавляется в  датафрейм
    с нормировкой на количество записей (rn_max) для каждого id.
    Удаляет уже не нужные колонки.

    Args:
        df: (pd.DataFrame)  Исходный DataFrame с признаками из columns_list.
        columns_list: (List[str]) Список столбцов, для которых считаем среднюю частоту значений
        drop_list(List[str], optional): Список уже не нужных признаков,
            для удаления из датафрейма

    Returns:
        pandas.DataFrame :  Копия DataFrame с добавленным новым столбцом {column}_mean_freq,
        содержащим нормированное агрегированное значение средней
        частоты значений column для каждого id.

    ВАЖНО: функция требует два аргумента на входе, что несовместимо с работой sklearn Pipeline
    (который ожидает функцию только с одним аргументом — DataFrame).
     Поэтому при добавлении этой функции в пайплайн её необходимо оборачивать с помощью partial,
      чтобы зафиксировать дополнительные параметры заранее.
    """

    logger.info("NEW features")

    for col in columns_list:
        new_col = f"{col}_mean_freq"
        logger.info(new_col)

        # Вычисляем относительную частоту каждого уникального значения в столбце
        bin_freq = df[col].value_counts(normalize=True).to_dict()

        # Создаём Series с частотами значений для каждой строки
        freq_series = df[col].map(bin_freq)
        """
        Делаем группировку столбца по id и считаем сумму частот в группе,
        делим сумму на количество записей для этого id.
        Результат сохраняем в новый столбец new_col.
        """
        df[new_col] = freq_series.groupby(df["id"]).transform("sum") / df["rn_max"]

        # Удаляем временные переменные для экономии памяти
        del freq_series, bin_freq
        gc.collect()

    # Если передан список колонок на удаление
    if drop_list is not None:
        # Удаляем уже не нужные колонки
        df = df.drop(drop_list, axis=1)
        logger.info(f"DataFrame shape after drop(): {df.shape}")

    return df


@memory_monitor_function
def definite_value_proportion_features_pipeline(
    df: pd.DataFrame,
    features_dictionary: Dict[str, Any],
    float_downcast_columns_list: List[str] = None,
) -> pd.DataFrame:
    """
    Создаёт и добавляет в датафрейм новые частотные признаки
    на основе заданных значений исходных признаков.

    Для каждого столбца и каждого указанного значения в словаре функция создаёт новые признаки,
    отражающие долю записей с этим значением относительно общего количества
    кредитов (rn_max) для каждого id.
    Меняет тип новых колонок с float64 на float32 согласно списку.
    Удаляет уже не нужные колонки.

    Args:
        df : Исходный DataFrame, содержащий необходимые признаки и колонку 'rn_max'.
        features_dictionary: Dict[str, Any] - Словарь где ключами являются названия колонок,
            а значениями уникальные значения колонки которые требуется обработать.
        float_downcast_columns_list (List[str], optional) : Список колонок  тип которых можно
            безопасно понизить с float64 до float32 без потери информативности
            из-за округления значений.

    Returns:
        pandas.DataFrame : Копия исходного DataFrame с добавленными частотными признаками.

    ВАЖНО: функция требует два аргумента на входе, что несовместимо с работой sklearn Pipeline
    (который ожидает функцию только с одним аргументом — DataFrame).
     Поэтому при добавлении этой функции в пайплайн её необходимо оборачивать с помощью partial,
      чтобы зафиксировать дополнительные параметры заранее.
    """

    # Итерируем по ключам
    for col in features_dictionary.keys():
        logger.debug("BEFORE column statistics")
        # Проверим RSS процесса и объекты в RAM
        rss_process_statistic(df)
        # Проверим потребление памяти по cgroup
        cgroup_memory_statistic()

        logger.info("Original feature %s", col)

        logger.info("New features")

        # Итерируем по значениям
        for value in features_dictionary[col]:
            new_col = f"{col}_prop_{value}"
            logger.info(new_col)

            """
            Создаём булевую маску: True, если значение в col равно value,
            иначе False.
            """
            mask = df[col] == value
            """
            Для каждой строки вычисляем количество совпадений value 
            по id (transform('sum')) и делим на общее количество кредитов 
            по id (rn_max), чтобы получить долю.
            """
            df[new_col] = mask.groupby(df["id"]).transform("sum") / df["rn_max"]

            # Выведем  тип новой колоноки
            logger.info(f"New column type is: {df[new_col].dtype}")

            #  Если передан список для смены типов колонок
            if float_downcast_columns_list is not None:
                # По условию меняем тип колонки с float64 на float32
                if new_col in float_downcast_columns_list:
                    df[new_col] = df[new_col].astype("float32")

            # Удаляем маску
            del mask
            gc.collect()

            # Выведем тип новой колоноки
            logger.info(f"After changing New column type is: {df[new_col].dtype}")

            # Выведем размер датафрейма
            logger.info(f"With New column DataFrame shape: {df.shape}")

        # Пробуем оптимизировать RSS
        heap_trim()

    return df


@memory_monitor_function
def from_is_zero_prop_1_create_sum_prop_1_feature_pipeline(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Вычисляет среднее значение признаков is_zero_*_prop_1 по строкам и добавляет
    новый признак 'is_zero_sum_prop_1' в DataFrame.

    Args:
        df :  Исходный DataFrame с признаками is_zero_*_prop_1.

    Returns:
        pandas.DataFrame : Копия DataFrame с добавленным признаком 'is_zero_sum_prop_1'.
    """

    columns = [
        "is_zero_loans5_prop_1",
        "is_zero_loans530_prop_1",
        "is_zero_loans3060_prop_1",
        "is_zero_loans6090_prop_1",
        "is_zero_loans90_prop_1",
    ]

    df["is_zero_sum_prop_1"] = df[columns].sum(axis=1) / 5

    return df


@memory_monitor_function
def pre_since_opened_sum_mean_repeated_pipeline(df: pd.DataFrame) -> pd.DataFrame:
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

    # Считаем количество каждого значения 'pre_since_opened' для каждого 'id'
    counts = df.groupby(["id", "pre_since_opened"]).size()

    """
    Оставляем только повторяющиеся значения (количество > 1), 
    вычитаем первое появление.
    """
    repeated_pre_since_opened = counts[counts > 1] - 1

    # Суммируем количество повторов по каждому 'id'
    sum_repeated = repeated_pre_since_opened.groupby("id").sum()

    # Добавляем отсутствующие 'id' с нулевыми значениями повторов
    all_sum_repeated = sum_repeated.reindex(df["id"].unique(), fill_value=0)

    # Добавляем новый столбец: для каждого 'id' записываем рассчитанную сумму повторов
    df["pre_since_opened_repeated_prop"] = df["id"].map(all_sum_repeated)

    # Удаляем временные переменные
    del (counts, repeated_pre_since_opened, sum_repeated, all_sum_repeated)
    gc.collect()

    # Нормируем сумму повторов на количество записей 'rn_max' для каждого 'id'
    df["pre_since_opened_repeated_prop"] = (
        df["pre_since_opened_repeated_prop"] / df["rn_max"]
    )

    # Понижаем тип новой колонки
    df["pre_since_opened_repeated_prop"] = df["pre_since_opened_repeated_prop"].astype(
        "float32"
    )

    return df


@memory_monitor_function
def drop_columns_pipeline(df: pd.DataFrame, columns_list: List[str]) -> pd.DataFrame:
    """
    Удаляет исходные и временные признаки из DataFrame.

    Args:
        df: (pd.DataFrame) Исходный DataFrame.
        columns_list: List[str] Список удаляемых колонок.

    Returns:
        pd.DataFrame : Копия DataFrame без указанных столбцов

    ВАЖНО: функция требует два аргумента на входе, что несовместимо с работой sklearn Pipeline
    (который ожидает функцию только с одним аргументом — DataFrame).
     Поэтому при добавлении этой функции в пайплайн её необходимо оборачивать с помощью partial,
      чтобы зафиксировать дополнительные параметры заранее.
    """

    # Выведем размер датафрейма
    logger.info(f"DataFrame shape: {df.shape}")
    # Сбрасываем ненужные признаки
    df = df.drop(columns_list, axis=1)
    logger.debug("Columns dropped successfully")
    # Выведем размер датафрейма
    logger.info(f"After dropping columns DataFrame shape : {df.shape}")

    return df


@memory_monitor_function
def drop_duplicates_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет  дубликаты по столбцу 'id', оставляя только первую запись.
    После удаления дубликатов столбец 'id' также удаляется.

    Args:
        df: (pd.DataFrame) Исходный DataFrame.

    Returns:
        pd.DataFrame : Копия DataFrame без дубликатов по 'id' и столбца 'id'.
    """

    # Выведем размер датафрейма
    logger.info(f"DataFrame shape: {df.shape}")

    # Удаляем дубликаты через группировку - оставляем первое появление каждого 'id'
    # Потребляет меньше RAM чем стандартный drop_duplicates()
    df = df.groupby("id", as_index=False).first()

    logger.debug("Duplicates dropped successfully")
    # Выведем размер датафрейма
    logger.info(f"DataFrame shape after dropping duplicates: {df.shape}")

    # Удаляем столбец 'id', так как он больше не нужен
    df = df.drop("id", axis=1)
    logger.debug("id dropped successfully")

    # Выведем размер датафрейма и типы колонок
    logger.info(f"DataFrame shape after dropping ID: {df.shape}")

    return df


# Добавим защитный блок main для тестов
if __name__ == "__main__":
    # Настройка логгера только для standalone запуска
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    pass
