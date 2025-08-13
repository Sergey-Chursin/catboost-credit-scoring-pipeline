import os
import glob
import datetime
import logging

from typing import List, Optional, Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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
        path_to_dataset: str,
        start_from: int = 0,
        num_parts_to_read: int = 1,
        verbose: bool = False,
        columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Читает указанные партиции Parquet из директории,
    преобразует их в pd.DataFrame и возвращает объединённый результат.

    Args:
        path_to_dataset : путь до директории с партициями
        start_from : номер партиции, с которой нужно начать чтение
        num_parts_to_read : количество партиций, которые требуется прочитать
        verbose : выводить ли дополнительную информацию
        columns : список колонок, которые нужно прочитать из партиции
             по умолчанию останутся все колонки


    Returns:
        pd.DataFrame
    """
    logger.info('Starting load_parquet_chunks function')

    res = []
    dataset_paths = sorted(
        os.path.join(path_to_dataset, filename)
        for filename in os.listdir(path_to_dataset)
        if filename.startswith('train')
    )
    logger.info(f'Found {len(dataset_paths)} dataset paths')

    start_from = max(0, start_from)
    chunks = dataset_paths[start_from: start_from + num_parts_to_read]

    logger.info('Reading chunks:')
    for chunk in chunks:
        logger.info(chunk)

    for chunk_path in tqdm(
            chunks,
            desc="Reading dataset with pandas",
            disable=not verbose, # бар отключится если verbose=False
            mininterval=5 # Обновление 1 раз в 5 сек
    ):
        logger.info(f'Reading chunk: {chunk_path}')

        chunk = pd.read_parquet(chunk_path, columns=columns)
        res.append(chunk)

    result = pd.concat(res).reset_index(drop=True)

    logger.info(f'Finished load_parquet_chunks (read {len(result)} rows)')

    return result

def load_dataset(
        path_to_dataset: str,
        num_parts_total: int,
        save_to_path: str,
        num_parts_to_preprocess_at_once: int = 1,
        verbose: bool = False,
        columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Загружает и подготавливает полный датасет из партиций Parquet,
     обрабатывает батчами,
     опционально сохраняет чанки и возвращает объединённый DataFrame.

    Args:
        path_to_dataset : путь до датасета с партициями
        num_parts_total : общее количество партиций, которые нужно обработать
        save_to_path : путь до папки для сохранения обработанных блоков в .parquet-формате;
            если None, сохранение не происходит
        num_parts_to_preprocess_at_once : количество партиций,
            которые будут одновременно держаться и обрабатываться в памяти
        verbose : логировать каждую обрабатываемую часть данных
        columns : список колонок, которые нужно оставить
            по умолчанию останутся все колонки

    Returns:
        pd.DataFrame : датафрейм с объединёнными данными
    """
    logger.info('Starting load_dataset function')

    preprocessed_frames = []

    # Добавлен disable=not verbose — бар отключится если verbose=False
    for step in tqdm(range(0, num_parts_total, num_parts_to_preprocess_at_once),
                     desc="Loading entire data",
                     disable=not verbose
                     ):
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

            logger.info(f'Saved to "{save_file}"')

        preprocessed_frames.append(transactions_frame)

    result = pd.concat(preprocessed_frames)

    logger.info(f'Finished load_dataset (total rows: {len(result)})')

    return result

def split_dataset_by_target(
        dataset: pd.DataFrame,
        path_to_target: str,
        train_size: float,
        random_state: int,
        stratify_col: str
) -> Dict[str, pd.DataFrame]:
    """
    Разделяет датасет на train/test на основе разделения
     стратифицированного разделения target.

    Args:
        dataset (pd.DataFrame): Входной датафрейм с признаками, без целевой переменной.
        path_to_target (str, optional): Путь к CSV-файлу с целевой переменной.
        train_size (float, optional): Доля обучающей выборки (от 0 до 1).
        random_state (int, optional): Значение random seed для воспроизводимости сплита.
        stratify_col (str, optional): Название колонки целевой переменной  для стратификации.

    Returns:
        Dict с 'X_train', 'y_train', 'X_test', 'y_test'
    """
    logger.info('Starting split_dataset_by_target')

    # Загружаем датасет с целевой переменной
    target = pd.read_csv(path_to_target)
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

    logger.info(f'Split completed:'
                    f' X_train {X_train.shape}'
                    f' X_test {X_test.shape}'
                    f' y_train {y_train.shape}'
                    f' y_test {y_test.shape}'
                    )

    return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}

def split_target_only(
        path_to_target: str,
        train_size: float,
        random_state: int,
        stratify_col: str,
        verbose: bool = False
):
    """
    Разделяет только таргет на train/test подвыборки.

    Args:
        path_to_target (str, optional): Путь к CSV-файлу с целевой переменной.
        train_size (float, optional): Доля train-выборки (от 0 до 1).
        random_state (int, optional): Значение random seed для воспроизводимости разбиения.
        stratify_col (str, optional): Название колонки для стратификации при сплите.
        verbose (bool, optional): Если True, выводит сообщения (print) о прогрессе в консоль.
            По умолчанию True.
    Returns:
        dict с pandas.Series: {'y_train', 'y_test'}
    """
    target = pd.read_csv(path_to_target)
    if verbose:
        print(f'Loaded target from {path_to_target}'
              f' (shape: {target.shape}'
              )

    y_train, y_test = train_test_split(
        target,
        train_size=train_size,
        random_state=random_state,
        stratify=target[stratify_col]
    )
    if verbose:
        print(
            f'y_train shape: {y_train.shape}\n'
            f'y_test shape: {y_test.shape}'
        )
    return {
        'y_train': y_train[stratify_col],
        'y_test': y_test[stratify_col]
    }

def make_file_path(
        output_type: str,
        data_path: str,
        output_dir: str,
        ext: str
) -> str:
    """
    Формирует путь для сохранения файла предсказаний с уникальным именем, включающим тип вывода,
    имя исходной папки с данными и текущую дату/время.

    Имя файла строится по шаблону:
    <output_type>__<имя_папки_источника>__<текущая_дата_и_время>.<ext>

    Args:
        output_type (str): Тип вывода (например, 'proba' или 'predict').
        data_path (str): Путь к исходной папке с данными, используется для извлечения имени.
        output_dir (str): Папка, в которую будет сохранён итоговый файл.
        ext (str): Расширение итогового файла (например, 'csv').

    Returns:
        str: Полный путь к файлу с предсказаниями.
    """
    # os.path.normpath(path) -приводит путь к "нормализованному" виду
    # (убирает лишние слэши, точки, двойные слэши и пр.)
    # os.path.basename(path) возвращает только "последнюю часть" пути — имя
    # файла или последней папки.
    base = os.path.basename(os.path.normpath(data_path))
    # Получаем текущее время
    dt = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    # Собираем имя файла
    filename = f"{output_type}__{base}__{dt}.{ext}"

    return os.path.join(output_dir, filename)


def check_data_folder_and_count_files(
    data_path: str,
    pattern: str,
) -> Tuple[List[str], int]:
    """
    Проверяет существование папки data_path и наличие файлов по маске (например, *.pq).
    Возвращает список путей к найденным файлам и их количество.

    Args:
        data_path (str): Путь к директории с исходными файлами.
        pattern (str): Маска расширения для поиска файлов.

    Returns:
        Tuple[List[str], int]: Список путей к найденным файлам и их количество.

    Raises:
        ValueError: Если директория отсутствует или не содержит файлов по маске.
    """
    logger.info(f"Starting check_data_folder_and_count_files : {data_path}")
    # Проверяем существование папки/файла, при отсутвии выводим предупреждение
    if not os.path.isdir(data_path):
        raise ValueError(f"Data path '{data_path}' is not a valid directory")

    # Определяем количество файлов в папке
    # glob.glob найдёт все файлы в папке по маске pattern ('*.pq')
    # Если файлов с таким расширением нет, выводим предупреждение
    file_paths = glob.glob(os.path.join(data_path, pattern))
    files_count = len(file_paths)
    if files_count == 0:
        raise ValueError(f"No files matching '{pattern}' in {data_path}")

    logger.info(f'Count of files in data folder: {files_count}')

    return file_paths, files_count



def save_predictions_with_id(
    output_type: str,
    ids: Union[np.ndarray, pd.Series, list],
    predictions: np.ndarray,
    output_path: str
):
    """
    Сохраняет предсказания (proba или predict) вместе с id в .csv
    Формат для proba поддерживает бинарный и многоклассовый формат:
        id, proba_class_0, proba_class_1, ...
    Формат для predict:
        id, prediction

    Args:
        output_type (str): 'proba' или 'predict'
        ids (array/Series/list): значения id для всех объектов
        predictions (np.ndarray): массив предиктов (classes или вероятности)
        output_path (str): итоговый путь к файлу .csv
    """
    logger.info("Started save_predictions_with_id function")
    df_pred = pd.DataFrame()
    df_pred['id'] = ids

    # Для вероятностей
    if output_type == 'proba':
        # Для вероятности одного класса
        # (бинарная после слайсинга [:, 1])
        if predictions.ndim == 1:
            df_pred['proba'] = predictions
        # Для бинарной и многоклассовой классификации
        elif predictions.ndim == 2:
            # Присвоим каждой вероятности свою колонку
            for i in range(predictions.shape[1]):
                df_pred[f'proba_class_{i}'] = predictions[:, i]
        else:
            raise ValueError(f"Unsupported predictions shape for proba: {predictions.shape}")
    # Для меток классов
    elif output_type == 'predict':
        df_pred['prediction'] = predictions
    else:
        raise ValueError(f"Unsupported output_type: {output_type}")

    df_pred.to_csv(output_path, index=False)


# Добавим защитный блок main для тестов
if __name__ == "__main__":
    # Настройка логгера только для standalone запуска
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s')

    pass