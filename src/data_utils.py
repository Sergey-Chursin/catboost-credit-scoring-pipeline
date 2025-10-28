import datetime
import gc
import glob
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

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


def cast_columns_by_map(
    df: pd.DataFrame, cast_type_map: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Меняет типы DataFrame-колонок в соответствии с заданным словарём.
    Колонки, отсутствующие в cast_type_map или не найденные в df, не изменяются.

    ВАЖНО:
    При попытке привести колонку с некорректными значениями (NaN, строки, неконвертируемые значения)
    тип этой колонки останется прежним, выполнение кода не прервётся, а предупреждение будет записано в лог.
    Все такие ошибочные значения и их обработка делегируются следующему этапу — preprocessing pipeline.

    Args:
        df (pd.DataFrame): Исходный DataFrame.
        cast_type_map (dict, optional): Словарь соответствий {имя_колонки(str): тип_данных(str)}.

    Returns:
        pd.DataFrame: DataFrame с приведёнными типами указанных колонок.
    """
    # Если словарь типов не задан — просто возвращаем исходный DataFrame
    if cast_type_map is None:
        return df

    for col, dtype in cast_type_map.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                logger.warning(f"Could not cast column '{col}' to {dtype}: {e}")
    return df


"""
Собираем исходный датасет из parquet файлов,  
скачиваем только необходимые колонки
"""


def load_data_chunks(
    path_to_dataset: str,
    start_from: int = 0,
    num_parts_to_read: int = 1,
    verbose: bool = False,
    columns: Optional[List[str]] = None,
    cast_type_map: Optional[dict] = None,
    mask: Optional[str] = None,
    search_file_ext: str = ".pq",
) -> pd.DataFrame:
    """
    Читает указанные партиции Parquet из директории, объединяет их в DataFrame
    и приводит типы указанных колонок.

    Args:
        path_to_dataset (str): Путь до директории с parquet-файлами.
        start_from (int, optional): Номер партиции, с которой начать чтение (по умолчанию 0).
        num_parts_to_read (int, optional): Количество партиций, которые требуется прочитать
            (по умолчанию 1).
        verbose (bool, optional): Если True, выводит выводит бары загрузки  файлов.
        columns (Optional[List[str]], optional): Список колонок, которые нужно прочитать из партиций
            (по умолчанию все).
        cast_type_map (Optional[dict], optional): Словарь {имя_колонки: тип}, где тип — строка для приведения типа
            (например, 'int8', 'float32', 'category'). Если None, типы не приводятся.
        mask (Optional[str], optional): Маска для выбора файлов в папке (например, 'train').
            Если указана, выбираются только файлы, имя которых начинается с mask;
            если None — выбираются все файлы.
        search_file_ext (str, optional): Расширение файлов для поиска: ".csv", ".pq", ".parquet").
            По умолчанию ".pq".

    Returns:
        pd.DataFrame: Объединённый DataFrame с выбранными колонками и приведёнными типами.
    """
    logger.info("Starting load_data_chunks function")

    # Список для накопления прочитанных DataFrame
    res = []

    # Собираем отсортированный список файлов с нужным расширением из папки:
    #  Если mask не задан (пустая строка), берём все файлы, заканчивающиеся на search_file_ext (например, '.parquet').
    # Если mask задан, берём только те файлы, которые начинаются с mask и заканчиваются на search_file_ext.
    # Это позволяет гибко отбирать либо все партиции указанного расширения, либо только конкретные
    # (например, 'train*.parquet', 'test*.pq') без риска схватить посторонние файлы.
    dataset_paths = sorted(
        os.path.join(path_to_dataset, filename)
        for filename in os.listdir(path_to_dataset)
        if filename.endswith(search_file_ext)
        and (not mask or filename.startswith(mask))
    )

    logger.info(f"Found {len(dataset_paths)} dataset paths")

    # Определяем диапазон файлов для чтения (батч)
    start_from = max(0, start_from)
    chunks = dataset_paths[start_from : start_from + num_parts_to_read]

    logger.info("Reading chunks:")
    for chunk in chunks:
        logger.info(chunk)

    # Выбираем функцию пандас для закачки файлов согласно переданному расширению
    ext_to_reader = {
        ".pq": lambda path, cols: pd.read_parquet(path, columns=cols),
        ".parquet": lambda path, cols: pd.read_parquet(path, columns=cols),
        ".csv": lambda path, cols: pd.read_csv(path, usecols=cols),
        # можно добавить другие форматы
    }
    reader = ext_to_reader[search_file_ext]

    # Читаем parquet-файлы по указанному батчу и накапливаем DataFrame'ы
    for chunk_path in tqdm(
        chunks,
        desc="Reading dataset with pandas",
        disable=not verbose,  # tqdm-бар выключен, если verbose=False
        mininterval=5,  # обновление прогресса раз в 5 секунд
    ):
        logger.info(f"Reading chunk: {chunk_path}")
        chunk = reader(path=chunk_path, cols=columns)
        res.append(chunk)

    # Объединяем все прочитанные DataFrame в один
    result = pd.concat(res).reset_index(drop=True)

    # Приводим колонки датафрейма к нужному типу, если словарь типов задан
    result = cast_columns_by_map(result, cast_type_map)

    logger.info(f"Finished load_data_chunks (read {len(result)} rows)")

    return result


def load_dataset(
    path_to_dataset: str,
    num_parts_total: int,
    save_to_path: Optional[str] = None,
    num_parts_to_preprocess_at_once: int = 1,
    verbose: bool = False,
    columns: Optional[List[str]] = None,
    cast_type_map: Optional[dict] = None,
    mask: Optional[str] = None,
    search_file_ext: str = ".pq",
) -> pd.DataFrame:
    """
    Обёртка для функции load_data_chunks.
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
        cast_type_map : Словарь {имя_колонки: тип},
            где тип — строка для приведения типа (например, 'int8', 'float32', 'category').
            Если None, типы не приводятся.
        mask (Optional[str], optional): Маска для выбора файлов в папке (например, 'train').
            Если указана, выбираются только файлы, имя которых начинается с mask;
            если None — выбираются все файлы.
        search_file_ext (str, optional): Расширение файлов для поиска (например, ".csv", ".pq").
            По умолчанию ".pq".

    Returns:
        pd.DataFrame : датафрейм с объединёнными данными
    """
    logger.info("Starting load_dataset function")

    # Финальный датафрейм для объединения всех частей
    result = None

    # tqdm организует прогресс-бар по всему процессу загрузки
    # disable=not verbose — бар отключится если verbose=False
    for step in tqdm(
        range(0, num_parts_total, num_parts_to_preprocess_at_once),
        desc="Loading entire data",
        disable=not verbose,
    ):
        logger.info(f"Processing step {step}")

        # Загружаем одну или несколько партиций (батч)
        # с помощью функции load_data_chunks
        transactions_frame = load_data_chunks(
            path_to_dataset,
            start_from=step,
            num_parts_to_read=num_parts_to_preprocess_at_once,
            verbose=verbose,
            columns=columns,
            cast_type_map=cast_type_map,
            mask=mask,
            search_file_ext=search_file_ext,
        )

        # Если указан путь для сохранения — сохраняем обработанный
        # батч в отдельный parquet-файл.
        # Мzfill - "заполняет" строку нулями слева до указанной длины
        if save_to_path:
            block_as_str = str(step).zfill(3)
            save_file = os.path.join(
                save_to_path, f"processed_chunk_{block_as_str}.parquet"
            )
            transactions_frame.to_parquet(save_file)

            logger.info(f'Saved to "{save_file}"')

        # Склеиваем датафреймы: если это первая часть — просто назначаем,
        # иначе склеиваем с предыдущими.
        if result is None:
            result = transactions_frame
        else:
            result = pd.concat([result, transactions_frame], ignore_index=True)

        # Освобождаем память за ненужный уже датафрейм батча
        del transactions_frame
        gc.collect()

    logger.info(f"Finished load_dataset (total rows: {len(result)})")

    return result


def split_dataset_by_target(
    dataset: pd.DataFrame,
    path_to_target: str,
    train_size: float,
    random_state: int,
    stratify_col: str,
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
    logger.info("Starting split_dataset_by_target")

    # Загружаем датасет с целевой переменной
    target = pd.read_csv(path_to_target)
    logger.info(f'Loaded target from "{path_to_target}" (shape: {target.shape})')

    # Делим датасет с целевой переменной на train/test части
    y_train, y_test = train_test_split(
        target,
        train_size=train_size,
        random_state=random_state,
        stratify=target[stratify_col],
    )

    # Отсортируем результат по id
    y_train = y_train.sort_values(by="id").reset_index(drop=True)
    y_test = y_test.sort_values(by="id").reset_index(drop=True)

    # Забираем наборы id из train/test
    train_id = y_train["id"].values
    test_id = y_test["id"].values

    # На основе наборов id делим исходный датасет на train/test части
    # сортируем для приведения к порядку идентичному с id таргета
    X_train = (
        dataset[dataset["id"].isin(train_id)]
        .sort_values(by="id")
        .reset_index(drop=True)
    )
    X_test = (
        dataset[dataset["id"].isin(test_id)].sort_values(by="id").reset_index(drop=True)
    )

    # Сбросим индексы наборов таргета для приведения к единому виду с X_train/X_test
    y_train = y_train.reset_index(drop=True)[stratify_col]
    y_test = y_test.reset_index(drop=True)[stratify_col]

    logger.info(
        f"Split completed:"
        f" X_train {X_train.shape}"
        f" X_test {X_test.shape}"
        f" y_train {y_train.shape}"
        f" y_test {y_test.shape}"
    )

    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}


def split_target_only(
    path_to_target: str,
    train_size: float,
    random_state: int,
    stratify_col: str,
    verbose: bool = False,
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
        print(f"Loaded target from {path_to_target} (shape: {target.shape}")

    y_train, y_test = train_test_split(
        target,
        train_size=train_size,
        random_state=random_state,
        stratify=target[stratify_col],
    )
    if verbose:
        print(f"y_train shape: {y_train.shape}\ny_test shape: {y_test.shape}")
    return {"y_train": y_train[stratify_col], "y_test": y_test[stratify_col]}


def make_file_path(output_type: str, data_path: str, output_dir: str, ext: str) -> str:
    """
    Формирует путь для сохранения файла предсказаний с уникальным именем, включающим тип вывода,
    имя исходной папки с данными и текущую дату/время.

    Имя файла строится по шаблону:
    <output_type>_<имя_папки_источника>_<текущая_дата_и_время>.<ext>

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
    filename = f"{output_type}_{base}_{dt}.{ext}"

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

    logger.info(f"Count of files in data folder: {files_count}")

    return file_paths, files_count


def save_predictions_with_id(
    output_type: str,
    ids: Union[np.ndarray, pd.Series, list],
    predictions: np.ndarray,
    output_path: str,
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
    df_pred["id"] = ids

    # Для вероятностей
    if output_type == "proba":
        # Для вероятности одного класса
        # (бинарная после слайсинга [:, 1])
        if predictions.ndim == 1:
            df_pred["proba"] = predictions
        # Для бинарной и многоклассовой классификации
        elif predictions.ndim == 2:
            # Присвоим каждой вероятности свою колонку
            for i in range(predictions.shape[1]):
                df_pred[f"proba_class_{i}"] = predictions[:, i]
        else:
            raise ValueError(
                f"Unsupported predictions shape for proba: {predictions.shape}"
            )
    # Для меток классов
    elif output_type == "predict":
        df_pred["prediction"] = predictions
    else:
        raise ValueError(f"Unsupported output_type: {output_type}")

    df_pred.to_csv(output_path, index=False)


# Добавим защитный блок main для тестов
if __name__ == "__main__":
    # Настройка логгера только для standalone запуска
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    pass
