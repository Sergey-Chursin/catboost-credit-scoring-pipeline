import logging
from typing import Any, List, Dict
import gc

import numpy as np
import pandas as pd

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
    Удаляет исходную колонку 'rn'.

    Args:
        df : Исходный DataFrame, содержащий колонки 'id' и 'rn'.

    Returns:
        pandas.DataFrame : Копия исходного DataFrame с добавленной колонкой 'rn_max'.
    """
    logger.info('FUNCTION rn_max_feature_pipeline')
    logger.info(f"DataFrame fragmentation: number of memory blocks = {df._mgr.nblocks}")

    # При передачи между функциями pipeline сильно фрагментирует датафрейм (разные memory blocks)
    # что существенно замедляет дальнейшие операции из-за внутренней структуры pandas.
    # Обычная копия (df.copy()) дефрагментирует объект.
    df = df.copy()
    logger.info(
        f"DataFrame fragmentation after copy(): number of memory blocks = {df._mgr.nblocks}"
    )
    # Для экономии памяти сборщик мусора сразу после копирования освободит ресурсы
    gc.collect()

    # ПРОВЕРКА ДАТАФРЕЙМОВ В ПАМЯТИ RAM
    for obj in gc.get_objects():
        if isinstance(obj, pd.DataFrame):
            logger.info(f'DF in RAM: {obj.shape}    {id(obj)}')

    # Для каждой строки определяем максимальное значение 'rn' среди всех строк с тем же 'id'
    #Метод transform('max') возвращает Series длины исходного DataFrame, где для каждой строки
    #указано максимальное значение 'rn' в её группе 'id'.

    df['rn_max'] = df.groupby('id')['rn'].transform('max')

    # Удаляем уже не нужный столбец для экономии памяти
    df = df.drop('rn', axis=1)

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
    logger.info(f"DataFrame fragmentation: number of memory blocks = {df._mgr.nblocks}")

    # При передачи между функциями pipeline сильно фрагментирует датафрейм (разные memory blocks)
    # что существенно замедляет дальнейшие операции из-за внутренней структуры pandas.
    # Обычная копия (df.copy()) дефрагментирует объект.
    df = df.copy()
    logger.info(
        f"DataFrame fragmentation after copy(): number of memory blocks = {df._mgr.nblocks}"
    )
    # Для экономии памяти сборщик мусора сразу после копирования освободит ресурсы
    gc.collect()

    # ПРОВЕРКА ДАТАФРЕЙМОВ В ПАМЯТИ RAM
    for obj in gc.get_objects():
        if isinstance(obj, pd.DataFrame):
            logger.info(f'DF in RAM: {obj.shape}    {id(obj)}')


    # Список колонок для перекодировки
    columns = [col for col in df.columns if col.startswith('enc_paym_')]

    # for col in columns:
        # Проверяем, есть ли значение 4 в колонке
        # if 4 in df[col].unique():
        #     # Заменяем значения согласно маппингу
        #     df.loc[:, col] = df[col].replace({1: 0, 2: 1, 3: 2, 4: 3})

    # Заменяем значения в любом случае, а не только если есть 4
    for col in columns:
        df.loc[:, col] = df[col].replace({1: 0, 2: 1, 3: 2, 4: 3})

    return df

def enc_paym_norm_group_sum_diff_pipeline(
        df: pd.DataFrame,
        drop_list: List[str]
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
    """

    logger.info('FUNCTION enc_paym_norm_group_sum_diff_pipeline')
    logger.info(f"DataFrame fragmentation: number of memory blocks = {df._mgr.nblocks}")

    # При передачи между функциями pipeline сильно фрагментирует датафрейм (разные memory blocks)
    # что существенно замедляет дальнейшие операции из-за внутренней структуры pandas.
    # Обычная копия (df.copy()) дефрагментирует объект.
    df = df.copy()
    logger.info(
        f"DataFrame fragmentation after copy(): number of memory blocks = {df._mgr.nblocks}"
    )
    # Для экономии памяти сборщик мусора сразу после копирования освободит ресурсы
    gc.collect()

    # ПРОВЕРКА ДАТАФРЕЙМОВ В ПАМЯТИ RAM
    for obj in gc.get_objects():
        if isinstance(obj, pd.DataFrame):
            logger.info(f'DF in RAM: {obj.shape}    {id(obj)}')


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

    # Удаляем временный df_buff
    del df_buff
    gc.collect()

    # ПРОВЕРКА ДАТАФРЕЙМОВ В ПАМЯТИ RAM
    for obj in gc.get_objects():
        if isinstance(obj, pd.DataFrame):
            logger.info(f'DF in RAM: {obj.shape}    {id(obj)}')

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

    # Выведем размер датафрейма и типы колонок
    logger.info(f"DataFrame shape: {df.shape}")
    for col, dtype in df.dtypes.items():
        logger.info(f"{col}: {dtype}")


    # Удаляем уже не нужные колонки
    df = df.drop(drop_list, axis=1)
    gc.collect()
    logger.info(f"DataFrame shape after drop(): {df.shape}")

    # ПРОВЕРКА ДАТАФРЕЙМОВ В ПАМЯТИ RAM
    for obj in gc.get_objects():
        if isinstance(obj, pd.DataFrame):
            logger.info(f'DF in RAM: {obj.shape}    {id(obj)}')

    return df


def mean_value_frequency_feature_pipeline(
        df: pd.DataFrame,
        columns_list: List[str],
        drop_list: List[str]
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
        drop_list: List[str]: Список уже не нужных признаков,
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
    logger.info('FUNCTION mean_value_frequency_feature_pipeline')
    logger.info(f"DataFrame fragmentation: number of memory blocks = {df._mgr.nblocks}")

    # При передачи между функциями pipeline сильно фрагментирует датафрейм (разные memory blocks)
    # что существенно замедляет дальнейшие операции из-за внутренней структуры pandas.
    # Обычная копия (df.copy()) дефрагментирует объект.
    df = df.copy()
    logger.info(
        f"DataFrame fragmentation after copy(): number of memory blocks = {df._mgr.nblocks}"
    )
    # Для экономии памяти сборщик мусора сразу после копирования освободит ресурсы
    gc.collect()

    # ПРОВЕРКА ДАТАФРЕЙМОВ В ПАМЯТИ RAM
    for obj in gc.get_objects():
        if isinstance(obj, pd.DataFrame):
            logger.info(f'DF in RAM: {obj.shape}    {id(obj)}')


    logger.info('New features')

    for col in columns_list:
        new_col = f'{col}_mean_freq'
        logger.info(new_col)

        # Вычисляем относительную частоту каждого уникального значения в столбце
        bin_freq = df[col].value_counts(normalize=True).to_dict()

        # Создаём Series с частотами значений для каждой строки
        freq_series = df[col].map(bin_freq)
        """
        Для каждой строки считаем сумму частот значений (freq_series) по группе 'id'.
        Делим эту сумму на общее количество записей по id (rn_max),
        чтобы получить нормированную среднюю частоту встречаемости значений
        признака для данного id.
        Результат сохраняем в новый столбец new_col.
        """
        df[new_col] = freq_series.groupby(df['id']).transform('sum') / df['rn_max']

        # Удаляем временные переменные, так как из за них
        # в RAM залипает копия датафрейма
        del freq_series, bin_freq
        gc.collect()

    # Выведем размер датафрейма и типы колонок
    logger.info(f"DataFrame shape: {df.shape}")
    # for col, dtype in df.dtypes.items():
    #     logger.info(f"{col}: {dtype}")

    # ПРОВЕРКА ДАТАФРЕЙМОВ В ПАМЯТИ RAM
    for obj in gc.get_objects():
        if isinstance(obj, pd.DataFrame):
            logger.info(f'DF in RAM: {obj.shape}    {id(obj)}')


    # ИЩЕТ ССЫЛКИ НА ОБЪЕКТ
    def debug_df_refs(df, tag=""):
        df_id = id(df)
        logger.info(f"{tag}: DataFrame id={df_id}, shape={df.shape}")
        referrers = gc.get_referrers(df)
        logger.info(f"{tag}: Number of referrers: {len(referrers)}")
        for i, ref in enumerate(referrers):
            logger.info(f"{tag}: Referrer {i}: {type(ref)} | Preview: {str(ref)[:500]}")


    logger.info("Drop columns")
    # Удаляем уже не нужные колонки
    df_new = df.drop(drop_list, axis=1)


    del df
    df = df_new.copy()
    del df_new

    gc.collect()


    debug_df_refs(df, "after drop")

    # ПРОВЕРКА ДАТАФРЕЙМОВ В ПАМЯТИ RAM
    for obj in gc.get_objects():
        if isinstance(obj, pd.DataFrame):
            logger.info(f'DF in RAM: {obj.shape}    {id(obj)}')

    # Выведем размер датафрейма и типы колонок
    logger.info(f"DataFrame shape: {df.shape}")
    for col, dtype in df.dtypes.items():
        logger.info(f"{col}: {dtype}")

    return df


def definite_value_proportion_features_pipeline(
        df: pd.DataFrame,
        features_dictionary: Dict[str, Any],
        drop_list: List[str],
        float_downcast_columns_list: List[str]
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
        drop_list: List[str] - Список уже не нужных признаков для удаления
        float_downcast_columns_list: List[str]: Список колонок  тип которых можно
            безопасно понизить с float64 до float32 без потери информативности
            из-за округления значений.

    Returns:
        pandas.DataFrame : Копия исходного DataFrame с добавленными частотными признаками.

    ВАЖНО: функция требует два аргумента на входе, что несовместимо с работой sklearn Pipeline
    (который ожидает функцию только с одним аргументом — DataFrame).
     Поэтому при добавлении этой функции в пайплайн её необходимо оборачивать с помощью partial,
      чтобы зафиксировать дополнительные параметры заранее.
    """
    logger.info('FUNCTION definite_value_proportion_features_pipeline')
    logger.info(f"DataFrame fragmentation: number of memory blocks = {df._mgr.nblocks}")

    # При передачи между функциями pipeline сильно фрагментирует датафрейм (разные memory blocks)
    # что существенно замедляет дальнейшие операции из-за внутренней структуры pandas.
    # Обычная копия (df.copy()) дефрагментирует объект.
    df = df.copy()
    logger.info(
        f"DataFrame fragmentation after copy(): number of memory blocks = {df._mgr.nblocks}"
    )
    # Для экономии памяти сборщик мусора сразу после копирования освободит ресурсы
    gc.collect()

    # ПРОВЕРКА ДАТАФРЕЙМОВ В ПАМЯТИ RAM
    for obj in gc.get_objects():
        if isinstance(obj, pd.DataFrame):
            logger.info(f'DF in RAM: {obj.shape}    {id(obj)}')

    """
    Создадим словарь где для каждого признака перечислены значения,
    по которым считаем долю.
    """

    # Итерируем по ключам
    for col in features_dictionary.keys():
        logger.info('Original feature %s', col)
        logger.info('New features')

        # Итерируем по значениям
        for value in features_dictionary[col]:
            new_col = f'{col}_prop_{value}'
            logger.info(new_col)

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
            df[new_col] = mask.groupby(df['id']).transform('sum') / df['rn_max']

            # Выведем  тип новой колоноки
            logger.info(f"New column type is: {df[new_col].dtype}")

            # Выводим вес датафрейма в RAM в мегабайтах
            mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
            logging.info(f"Before type changing DataFrame size in memory: {mem_mb:.2f} MB")

            # По условию меняем тип колонки с float64 на float32
            if new_col in float_downcast_columns_list:
                df[new_col] = df[new_col].astype('float32')

            # Удаляем маску
            del mask
            gc.collect()

            # Выведем размер  тип новой колоноки
            logger.info(f"After changing New column type is: {df[new_col].dtype}")

            # Выведем размер датафрейма
            logger.info(f"DataFrame shape: {df.shape}")

            # Выводим вес датафрейма в RAM в мегабайтах
            mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
            logging.info(f"After type changing DataFrame size in memory: {mem_mb:.2f} MB")

            # Выодим фрагментацию датафрейма
            logger.info(
                f"DataFrame fragmentation BEFORE copy(): number of memory blocks = {df._mgr.nblocks}"
            )
            # Дефрагментируем датафрейм
            logger.info("Defragmentation")
            df = df.copy()
            gc.collect()

            # ПРОВЕРКА ДАТАФРЕЙМОВ В ПАМЯТИ RAM
            for obj in gc.get_objects():
                if isinstance(obj, pd.DataFrame):
                    logger.info(f'DF in RAM: {obj.shape}    {id(obj)}')

            # Выодим фрагментацию датафрейма
            logger.info(
                f"DataFrame fragmentation AFTER copy(): number of memory blocks = {df._mgr.nblocks}"
            )

            # Выводим вес датафрейма в RAM в мегабайтах
            mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
            logging.info(f"DataFrame size in memory AFTER copy(): {mem_mb:.2f} MB")

            # ПРОВЕРКА ДАТАФРЕЙМОВ В ПАМЯТИ RAM
            for obj in gc.get_objects():
                if isinstance(obj, pd.DataFrame):
                    logger.info(f'DF in RAM: {obj.shape}    {id(obj)}')

            # ПРОВЕРКА SIRIESES В ПАМЯТИ RAM
            for obj in gc.get_objects():
                if isinstance(obj, pd.Series):
                    logger.info(f"Series in memory: {obj.name}  {id(obj)}")


        # Выводим вес датафрейма в RAM в мегабайтах
        mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
        logging.info(f"DataFrame size in memory BEFORE dropping old col: {mem_mb:.2f} MB")

        if col in drop_list:
            df = df.drop(col, axis=1)

            # Выводим вес датафрейма в RAM в мегабайтах
            mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
            logging.info(f"DataFrame size in memory AFTER dropping old col: {mem_mb:.2f} MB")

            gc.collect()

    # Выведем размер датафрейма и типы колонок
    logger.info(f"DataFrame shape: {df.shape}")
    for col, dtype in df.dtypes.items():
        logger.info(f"{col}: {dtype}")

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
    logger.info(f"DataFrame fragmentation: number of memory blocks = {df._mgr.nblocks}")

    # При передачи между функциями pipeline сильно фрагментирует датафрейм (разные memory blocks)
    # что существенно замедляет дальнейшие операции из-за внутренней структуры pandas.
    # Обычная копия (df.copy()) дефрагментирует объект.
    df = df.copy()
    logger.info(
        f"DataFrame fragmentation after copy(): number of memory blocks = {df._mgr.nblocks}"
    )
    # Для экономии памяти сборщик мусора сразу после копирования освободит ресурсы
    gc.collect()

    # Выведем размер датафрейма и типы колонок
    logger.info(f"DataFrame shape: {df.shape}")
    for col, dtype in df.dtypes.items():
        logger.info(f"{col}: {dtype}")


    columns = [
        'is_zero_loans5_prop_1',
        'is_zero_loans530_prop_1',
        'is_zero_loans3060_prop_1',
        'is_zero_loans6090_prop_1',
        'is_zero_loans90_prop_1'
    ]

    df['is_zero_sum_prop_1'] = df[columns].sum(axis=1) / 5

    # Выведем размер датафрейма и типы колонок
    logger.info(f"DataFrame shape: {df.shape}")
    for col, dtype in df.dtypes.items():
        logger.info(f"{col}: {dtype}")

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
    logger.info(f"DataFrame fragmentation: number of memory blocks = {df._mgr.nblocks}")

    # При передачи между функциями pipeline сильно фрагментирует датафрейм (разные memory blocks)
    # что существенно замедляет дальнейшие операции из-за внутренней структуры pandas.
    # Обычная копия (df.copy()) дефрагментирует объект.
    df = df.copy()
    logger.info(
        f"DataFrame fragmentation after copy(): number of memory blocks = {df._mgr.nblocks}"
    )
    # Для экономии памяти сборщик мусора сразу после копирования освободит ресурсы
    gc.collect()


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

    # Понижаем тип новой колонки
    df['pre_since_opened_repeated_prop'] = (
        df['pre_since_opened_repeated_prop'].astype('float32')
    )
    # Выведем размер датафрейма и типы колонок
    logger.info(f"DataFrame shape: {df.shape}")
    for col, dtype in df.dtypes.items():
        logger.info(f"{col}: {dtype}")

    return df


def drop_columns_drop_duplicates_pipeline(
        df: pd.DataFrame,
        columns_list: List[str]
) -> pd.DataFrame:
    """
    Удаляет исходные и временные признаки из DataFrame,
    а также удаляет дубликаты по столбцу 'id', оставляя только первую запись.
    После удаления дубликатов столбец 'id' также удаляется.

    Args:
        df: (pd.DataFrame) Исходный DataFrame.
        columns_list: List[str] Список удаляемых колонок.

    Returns:
        pd.DataFrame : Копия DataFrame без указанных столбцов и дубликатов по 'id'.

    ВАЖНО: функция требует два аргумента на входе, что несовместимо с работой sklearn Pipeline
    (который ожидает функцию только с одним аргументом — DataFrame).
     Поэтому при добавлении этой функции в пайплайн её необходимо оборачивать с помощью partial,
      чтобы зафиксировать дополнительные параметры заранее.
    """

    logger.info('FUNCTION drop_columns_drop_duplicates_pipeline')
    logger.info(f"DataFrame fragmentation: number of memory blocks = {df._mgr.nblocks}")

    # При передачи между функциями pipeline сильно фрагментирует датафрейм (разные memory blocks)
    # что существенно замедляет дальнейшие операции из-за внутренней структуры pandas.
    # Обычная копия (df.copy()) дефрагментирует объект.
    df = df.copy()
    logger.info(
        f"DataFrame fragmentation after copy(): number of memory blocks = {df._mgr.nblocks}"
    )
    # Для экономии памяти сборщик мусора сразу после копирования освободит ресурсы
    gc.collect()


    df = df.drop(columns_list, axis=1)
    # Явно вызываем сборщик мусора для освобождения памяти,
    # используемой предыдущими объектами DataFrame
    gc.collect()

    # Посчитаем и выведем в лог количество дубликатов по id
    n_dupes = df.duplicated().sum()
    logger.info(f" {n_dupes} duplicate records found")

    # Удаляем дубликаты по столбцу 'id', оставляя первую запись
    # и сразу сбрасываем индекс это экономит немного памяти.
    df = df.drop_duplicates(subset=['id'], keep='first', ignore_index=True)

    # Явно вызываем сборщик мусора для освобождения памяти,
    # используемой предыдущими объектами DataFrame
    gc.collect()

    # Выведем размер датафрейма и типы колонок
    logger.info(f"DataFrame shape: {df.shape}")
    for col, dtype in df.dtypes.items():
        logger.info(f"{col}: {dtype}")

    # Удаляем столбец 'id', так как он больше не нужен
    df = df.drop('id', axis=1)

    # Явно вызываем сборщик мусора для освобождения памяти,
    # используемой предыдущими объектами DataFrame
    gc.collect()

    # Выведем размер датафрейма и типы колонок
    logger.info(f"DataFrame shape: {df.shape}")
    for col, dtype in df.dtypes.items():
        logger.info(f"{col}: {dtype}")

    return df

# Добавим защитный блок main для тестов
if __name__ == "__main__":
    # Настройка логгера только для standalone запуска
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s')

    pass