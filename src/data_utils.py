import os
import glob
import logging
from typing import List, Optional, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (
    # Для load_dataset
    RAW_DATA_PATH,
    TEMP_DATA_PATH,
    PRE_FEATURES,
    NUM_PARTS_TOTAL,
    # Для split_dataset_by_target
    TARGET_PATH,
    TRAIN_SIZE,
    SEED_SPLIT_DATASET,
    STRATIFY_COL
)
"""
Создаём локальный логгер для этого модуля
Он наследует настройки от root logger
импортирующего файла (pipeline.py)
"""
logger = logging.getLogger(__name__)

"""
Собираем исходный датасет из parquet файлов,  
скачиваем только необходимые колонки
"""
def load_parquet_chunks(
        path_to_dataset: str = RAW_DATA_PATH,
        start_from: int = 0,
        num_parts_to_read: int = 1,
        columns: Optional[List[str]] = PRE_FEATURES,
        verbose: bool = False
) -> pd.DataFrame:
    """
    Читает указанные партиции Parquet из директории,
    преобразует их в pd.DataFrame и возвращает объединённый результат.

    Args:
        path_to_dataset : путь до директории с партициями
        start_from : номер партиции, с которой нужно начать чтение
        num_parts_to_read : количество партиций, которые требуется прочитать
        columns : список колонок, которые нужно прочитать из партиции
        verbose : выводить ли дополнительную информацию

    Returns:
        pd.DataFrame
    """
    if verbose:
        logger.info('Starting load_parquet_chunks function')

    res = []
    dataset_paths = sorted(
        os.path.join(path_to_dataset, filename)
        for filename in os.listdir(path_to_dataset)
        if filename.startswith('train')
    )
    if verbose:
        logger.info(f'Found {len(dataset_paths)} dataset paths')

    start_from = max(0, start_from)
    chunks = dataset_paths[start_from: start_from + num_parts_to_read]

    if verbose:
        logger.info('Reading chunks:')
        for chunk in chunks:
            logger.info(chunk)

    for chunk_path in tqdm(
            chunks,
            desc="Reading dataset with pandas",
            disable=not verbose, # бар отключится если verbose=False
            mininterval=5 # Обновление 1 раз в 5 сек
    ):
        if verbose:
            logger.info(f'Reading chunk: {chunk_path}')
        chunk = pd.read_parquet(chunk_path, columns=columns)
        res.append(chunk)

    result = pd.concat(res).reset_index(drop=True)
    if verbose:
        logger.info(f'Finished load_parquet_chunks (read {len(result)} rows)')

    return result

def load_dataset(
        path_to_dataset: str = RAW_DATA_PATH,
        num_parts_to_preprocess_at_once: int = 1,
        num_parts_total: int = NUM_PARTS_TOTAL,
        save_to_path: str = TEMP_DATA_PATH,
        verbose: bool = False,
        columns: Optional[List[str]] = PRE_FEATURES
) -> pd.DataFrame:
    """
    Загружает и подготавливает полный датасет из партиций Parquet,
     обрабатывает батчами,
     опционально сохраняет чанки и возвращает объединённый DataFrame.

    Args:
        path_to_dataset : путь до датасета с партициями
        num_parts_to_preprocess_at_once : количество партиций,
            которые будут одновременно держаться и обрабатываться в памяти
        num_parts_total : общее количество партиций, которые нужно обработать
        save_to_path : путь до папки для сохранения обработанных блоков в .parquet-формате;
            если None, сохранение не происходит
        verbose : логировать каждую обрабатываемую часть данных
        columns : список колонок, которые нужно оставить

    Returns:
        pd.DataFrame : датафрейм с объединёнными данными
    """
    if verbose:
        logger.info('Starting load_dataset function')

    preprocessed_frames = []

    # Добавлен disable=not verbose — бар отключится если verbose=False
    for step in tqdm(range(0, num_parts_total, num_parts_to_preprocess_at_once),
                     desc="Loading entire data",
                     disable=not verbose
                     ):
        if verbose:
            logger.info(f'Processing step {step}')

        transactions_frame = load_parquet_chunks(
            path_to_dataset,
            start_from=step,
            num_parts_to_read=num_parts_to_preprocess_at_once,
            verbose=verbose,
            columns=columns
        )

        # Записываем подготовленные данные в файл
        # Меняем if-else на zfill - "заполняет" строку нулями слева до указанной длины
        if save_to_path:
            block_as_str = str(step).zfill(3)
            save_file = os.path.join(save_to_path, f'processed_chunk_{block_as_str}.parquet')
            transactions_frame.to_parquet(save_file)
            if verbose:
                logger.info(f'Saved to "{save_file}"')

        preprocessed_frames.append(transactions_frame)

    result = pd.concat(preprocessed_frames)
    if verbose:
        logger.info(f'Finished load_dataset (total rows: {len(result)})')
    return result


def split_dataset_by_target(
        dataset: pd.DataFrame,
        path_to_target: str = TARGET_PATH,
        train_size: float = TRAIN_SIZE,
        random_state: int = SEED_SPLIT_DATASET,
        stratify_col: str = STRATIFY_COL,
        verbose: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Разделяет датасет на train/test на основе разделения
     стратифицированного разделения target.

    Args:
        dataset: Основной DataFrame (из load_dataset)
        path_to_target: Путь к target.csv (default из config)
        train_size: Доля train (default из config)
        random_state: Seed (default из config)
        stratify_col: Колонка для стратификации (default из config)
        verbose: Включить логи
            Default False, приоритет у переданного из pipeline

    Returns:
        Dict с 'X_train', 'y_train', 'X_test', 'y_test'
    """
    if verbose:
        logger.info('Starting split_dataset_by_target')

    # Загружаем датасет с целевой переменной
    target = pd.read_csv(path_to_target)
    if verbose:
        logger.info(f'Loaded target from "{path_to_target}" (shape: {target.shape})')

    # Делим датасет с целевой переменной на train/test части
    y_train, y_test = train_test_split(
        target,
        train_size=train_size,
        random_state=random_state,
        stratify=target[stratify_col])

    # Забираем наборы id из train/test
    train_id = y_train['id'].values
    test_id = y_test['id'].values

    # На основе наборов id делим исходный датасет на train/test части
    X_train = dataset.set_index('id').loc[train_id].reset_index()
    X_test = dataset.set_index('id').loc[test_id].reset_index()

    # Сбросим индексы для приведения к единому виду с X_train/X_test
    y_train = y_train.reset_index(drop=True)[stratify_col]
    y_test = y_test.reset_index(drop=True)[stratify_col]

    if verbose:
        logger.info(f'Split completed:'
                    f' X_train {X_train.shape}'
                    f' X_test {X_test.shape}'
                    f' y_train {y_train.shape}'
                    f' y_test {y_test.shape}'
                    )

    return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}


def check_data_folder_and_count_files(
    data_path: str,
    pattern: str = '*.pq'
) -> Tuple[List[str], int]:
    """
    Проверяет существование папки data_path и наличие файлов по маске (например, *.pq).
    Возвращает список путей к найденным файлам и их количество.

    Args:
        data_path (str): Путь к директории с исходными файлами.
        pattern (str): Маска для поиска файлов (по умолчанию '*.pq').

    Returns:
        Tuple[List[str], int]: Список путей к найденным файлам и их количество.

    Raises:
        ValueError: Если директория отсутствует или не содержит файлов по маске.
    """
    logger.info("Starting check_data_folder_and_count_files")
    # Проверяем существование папки/файла, при отсутвии выводим предупреждение
    if not os.path.isdir(data_path):
        raise ValueError(f"Data path '{data_path}' is not a valid directory")

    # Определяем количество файлов в папке
    # glob.glob найдёт все файлы в папке по маске pattern ('*.pq')
    # Если файлов с таким расширением нет, выводим предупреждение
    files = glob.glob(os.path.join(data_path, pattern))
    files_count = len(files)
    if files_count == 0:
        raise ValueError(f"No files matching '{pattern}' in {data_path}")

    return files, files_count


# Добавим защитный блок main для тестов
if __name__ == "__main__":
    # Настройка логгера только для standalone запуска
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s')

    pass