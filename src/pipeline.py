import logging
import argparse
import pickle
import gc
import datetime
from typing import Any, Optional, List, Dict
from functools import partial

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from config import (
    SAMPLE_FRAC,
    PIPELINE_PATH,
    PARAMS_LIST,
    WEIGHTS_LIST,
    INFERENCE_OUTPUT_DIR,
    PRE_FEATURES,
    RAW_DATA_PATH,
    TEMP_DATA_PATH,
    TARGET_PATH,
    TRAIN_SIZE,
    SEED_SPLIT_DATASET,
    STRATIFY_COL,
    TEST_PREDICT_PATH,
    THRESHOLD,
    CAT_FEATURES,
    N_SPLITS,
    SEED,
    SHUFFLE,
    PROP_FEATURES_DICT,
    MEAN_FREQ_SOURCE_LIST,
    DROP_LIST,
    PARQUET_FILE_PATTERN,
    CLASSES_METRIC_LIST,
    SAVE_FILE_EXTENSION,
    CAST_TYPE_MAP,
    SEARCH_FILE_EXTENSION,
    DROP_LIST_ENC_PAYM_NORM_GROUP_SUMM_DIFF,
    DROP_LIST_MEAN_VALUE_FREQUENCY_FEATURE,
    FLOAT_DOWNCAST_COLUMNS_LIST,
    TRANSFORM_DATA_PATH
)

from data_utils import (
    load_dataset,
    split_dataset_by_target,
    check_data_folder_and_count_files,
    make_file_path,
    save_predictions_with_id
)

from memory_utils import memory_checkpoint

from evaluate_metrics import compute_and_log_metrics

from log_config import setup_logging

from preprocessing import (
    SampleMedianImputer,
    convert_all_to_numeric_preprocessing,
    cast_columns_by_map_preprocessing,
    drop_duplicates_preprocessing
)

from feature_engineering import (
    rn_max_feature_pipeline,
    enc_paym_transcoding_pipeline,
    definite_value_proportion_features_pipeline,
    from_is_zero_prop_1_create_sum_prop_1_feature_pipeline,
    mean_value_frequency_feature_pipeline,
    enc_paym_norm_group_sum_diff_pipeline,
    pre_since_opened_sum_mean_repeated_pipeline,
    drop_columns_pipeline,
    drop_duplicates_pipeline
)

from classifier import CatBoostEnsembleClassifier

"""
Настраиваем парсер аргументов для CLI-запуска.
Это позволяет запускать скрипт с флагами:

--help - Вывод help-сообщения

-----------------
--log-level info     рабочие логи: старт/завершение функций, основные операции, ошибки.

--log-level debug    диагностика: метрики памяти, детали выполнения, отладочная информация.
-----------------
--mode train             для загрузки тренировочного датасета, 
                         разделения его на train/test,
                         обучения пайплайна,
                         сохранения пайплайна.
                                   
--mode test              для загрузки тренировочного датасета, 
                         разделения его на train/test,
                         загрузки пайплайна,
                         получения и сохранения предикта. 
                                  
--mode inference         для загрузки  датасета из указанной папки
                         (по умолчанию это учебные данные),
                         получения и сохранения предикта.
                         Это имитирует получение предикта 
                         на новых данных.
 
--mode transform_split  для трансформации и сохранения тренировочного
                        или тестового набора данных. Набор выбирается
                        флагом  --transform-subset, без этого флага
                        трансформируется тренировочный набор.
          
--mode transform_data   для трансформации нового набора данных.         
------------------
--transform-subset train  выбор тренировочного набора данных для трансформации.  

--transform-subset test   выбор тестового набора для данных для трансформации.                      
------------------                 
--output proba       для режимов test/inference 
                     получение предикта вероятностей классов.
                    
--output predict     для режимов test/inference 
                     получение предикта меток классов.
------------------                
--data-path          путь к данным для inference и transform_data,
                     по умолчанию это путь
                     к тренировочному датасету.   
------------------                  
--eval-metrics off  нет вывода метрик на тестовой выборке. 

--eval-metrics auc  на тестовой выборки считается AUC SCORE
                    и выводится в логи.
                     
--eval-metrics acc  на тестовой выборки считается ACCURACY
                    и выводится в логи.   
------------------                    
--output-dir str    путь сохранения предиктов на новых данных,
                    по умолчанию /../predictions/inference/
------------------       
--max-files         количество файлов скачиваемое из указанной папки в режимах   
                    inference и transform_data. Без ввода этого флага
                    скачиваются все файлы. 
------------------                                                        
Флаги можно ставить в любом порядке.
Любое сочетание флагов  не вызовет ошибки,
если действие включаемое флагом не поддерживается в данном режиме
оно просто игнорируется. 
"""

# Создаём парсер c описанием для --help и
# форматированием многострочных help
parser = argparse.ArgumentParser(
    description='Launching the pipeline',
    formatter_class=argparse.RawTextHelpFormatter
)
"""
Настраиваем аргументы парсера 
choices - автоматически проверит ввод и выдаст ошибку при неверном значении
help - текст сообщения команды --help  
default - значение по умолчанию.
"""
# Режим логирования. По умолчанию логи отключены.
parser.add_argument(
    '--log-level',
    type=str,
    default='off',
    choices=['info', 'debug', 'off'],
    help=(
        'Logging level: \n'
        'info - enable detailed logs\n'
        'debug - add diagnostics logs\n'
        'off - disable logs\n'
        'Default: off\n'
        'Example: --log-level info'
    )
)
# Режима пайплайна. По умолчанию пайплайн обучается.
parser.add_argument(
    '--mode',
    type=str,
    default='train',
    choices=[
        'train',
        'test',
        'inference',
        'transform_split',
        'transform_data'
    ],
    help=(
        'Execution mode:\n'
        'train - fit and save model\n'
        'test - validate on the test set\n'
        'inference - predict on new data\n'
        'transform_split - split dataset, transform and save train/test set\n'
        'transform_data - transform and save new data\n'
        'Default: train\n'
        'Example: --mode test'
    )
)
# Выбор train/test набора в transform_split режиме
parser.add_argument(
    '--transform-subset',
    type=str,
    default='train',
    choices=['train', 'test'],
    help=(
        'Which subset to transform in transform_split mode:\n'
        'train - transform training subset\n'
        'test - transform test subset\n'
        'Default: train\n'
        'Example: --transform-subset test'
    )
)
# Режима вывода предикта. По умолчанию получаем вероятности классов.
parser.add_argument(
    '--output',
    type=str,
    default='proba',
    choices=['proba', 'predict'],
    help=(
        'Output type for test/inference:\n'
        'proba -only predicted probabilities\n'
        'predict - only predicted classes\n'
        'Default: proba\n'
        'Example: --output predict'
    )
)
# Переключатель оценки метрики (AUC/ACC)
parser.add_argument(
    "--eval-metrics",
    type=str,
    choices=["off", "auc", "acc"],
    default="off",
    help=(
        'Evaluate metrics after train/test/inference:\n'
        'auc  - calculate and print ROC AUC score\n'
        'acc  - calculate and print Accuracy score\n'
        'off - do not calculate metrics\n'
        'Default: off\n'
        'Example: --eval-metrics auc'
    )
)
# Путь к новым данным для режимов inference и transform_data
parser.add_argument(
    '--data-path',
    type=str,
    default=RAW_DATA_PATH,
    help=(
        'Path to data folder containing .pq (Parquet) files.\n'
        'This is used for  inference and transform_data modes to specify new data.\n'
        'In train/test modes, it is ignored — fixed paths from config are used instead.\n'
        'The script loads and concatenates all .pq files in the folder.\n'
        'Default: ../data/train/ (training data path).\n'
        'Example (for inference and transform_data): --data-path /path/to/new_data/'
    )
)
# Количество скачиваемых из папки файлов для режимов inference и transform_data
parser.add_argument(
    '--max-files',
    type=int,
    default=None,
    help=(
        'Maximum number of data files to process.\n'
        'Used for memory optimization in inference and transform_data modes.\n'
        'Default: process all files in the folder\n'
        'Example: --max-files 50'
    )
)
# Путь сохранения предиктов новых данных
parser.add_argument(
    '--output-dir',
    type=str,
    default=INFERENCE_OUTPUT_DIR,
    help=(
        'Path to folder where to save inference predictions.\n'
        'Default: INFERENCE_OUTPUT_DIR (../predictions/inference/)'
    )
)

# Соберём пайплайн обработки данных и обучения ансамбля моделей
def main_pipeline(
        sample_frac: float,
        params_list: List[Dict],
        weights_list: List[float],
        threshold: float,
        cat_features: List[str],
        n_splits: int,
        seed: int,
        shuffle: bool,
        drop_list_enc_paym_norm_summ_diff: List[str],
        mean_freq_source_list: List[str],
        drop_list_mean_value_frequency_feature: List[str],
        prop_features_dict: Dict[str, Any],
        float_downcast_columns_list: List[str],
        drop_list: List[str],
        cast_type_map: Dict[str, str],
        logger: Optional[logging.Logger] = None,

):
    """
    Создаёт и возвращает основной Pipeline для обучения и предсказания.

    Pipeline включает стандартные этапы препроцессинга и feature engineering
     над табличными данными,
    а также кастомный классификатор на основе ансамбля моделей CatBoost.

    Args:
        sample_frac (float): Доля строк исходных данных, используемая для вычисления медиан в SampleMedianImputer.
        params_list (List[Dict]): Список словарей с гиперпараметрами для каждой модели CatBoost в ансамбле
            (N фолдов + 1 финальная модель).
        weights_list (List[float]): Список весов для взвешивания предсказаний ансамбля моделей.
        threshold (float): Порог для жёсткой классификации (в CatBoostEnsembleClassifier, параметр predict).
        cat_features (List[str]): Список названий категориальных фичей для CatBoost.
        n_splits (int): Количество фолдов для ансамблирования моделей (StratifiedKFold).
        seed (int): Seed для воспроизводимости разбиения и обучения моделей.
        shuffle (bool): Флаг перемешивания данных при разбиении на фолды.
        drop_list_enc_paym_norm_summ_diff: List[str]: Список колонок на удаление в функции
            enc_paym_norm_group_sum_diff_pipeline.
        mean_freq_source_list (List[str]): Список признаков для расчёта средних частот значений в функции
            mean_value_frequency_feature_pipeline.
        drop_list_mean_value_frequency_feature: List[str]: Список колонок на удаление в функции
            mean_value_frequency_feature_pipeline.
        prop_features_dict (Dict[str, Any]): Словарь, определяющий признаки и значения для создания
            пропорциональных фичей в функции definite_value_proportion_features_pipeline.
        float_downcast_columns_list: List[str]: Список колонок  тип которых можно
            безопасно понизить с float64 до float32 без потери информативности
            из-за округления значений.
        drop_list (List[str]): Список признаков для удаления из датасета на последнем этапе пайплайна
        cast_type_map : dict  Словарь соответствия для приведения типов колонок при загрузке данных
            {имя_колонки(str): тип(str)}.
        logger (Optional[logging.Logger], default=None): Логгер для сообщений.
            Если None (по умолчанию), логирование этапов данной функции будет отключено.

    Returns:
        sklearn.pipeline.Pipeline: Собранный pipeline, готовый для обучения (fit).
    """

    # Создаём SampleMedianImputer для заполнения пустых значений медианами
    imputer = SampleMedianImputer(sample_frac=sample_frac)

    # Создаём пайплайн препроцессинга данных
    preprocessing_pipe = Pipeline(
        [
            (
                'to_numeric',
                FunctionTransformer(
                    convert_all_to_numeric_preprocessing
                )
            ),
            (
                'imputer', imputer
            ),
            (
                'memory_checkpoint_1',
                FunctionTransformer(
                    memory_checkpoint
                )
            ),
            (
                'cast_type',
                FunctionTransformer(
                    partial(
                        cast_columns_by_map_preprocessing,
                        cast_type_map=cast_type_map
                )
            )
            ),
            (
                'memory_checkpoint_2',
                FunctionTransformer(
                    memory_checkpoint
                )
            ),
            (
                'drop_duplicates_preprocessing',
                FunctionTransformer(
                    drop_duplicates_preprocessing
                )
            ),
            (
                'memory_checkpoint_3',
                FunctionTransformer(
                    memory_checkpoint
                )
            )
        ]
    )

    # Создадим объект классификатора
    classifier = CatBoostEnsembleClassifier(
        params_list=params_list,
        weights_list=weights_list,
        threshold=threshold,
        cat_features=cat_features,
        n_splits=n_splits,
        seed=seed,
        shuffle=shuffle,
        logger=logger
    )
    # Создаём основной пайплайн
    main_pipe = Pipeline(
        [
            (
                'preprocessing',
                preprocessing_pipe
            ),
            (
                'create_rn_max_feature',
                FunctionTransformer(
                    rn_max_feature_pipeline
                )
            ),

            (
                'enc_paym_transcoding',
                FunctionTransformer(
                    enc_paym_transcoding_pipeline
                )
            ),
            (
                'from_enc_paym_create_normalized_group_sum_features_then_diff_features',
                FunctionTransformer(
                    partial(
                        enc_paym_norm_group_sum_diff_pipeline,
                        drop_list=drop_list_enc_paym_norm_summ_diff
                    )
                )
            ),
            (
                'create_mean_value_frequency_feature',
                FunctionTransformer(
                    partial(
                        mean_value_frequency_feature_pipeline,
                        columns_list=mean_freq_source_list,
                        drop_list=drop_list_mean_value_frequency_feature
                    )
                )
            ),
            (
                'memory_checkpoint_4',
                FunctionTransformer(
                    memory_checkpoint
                )
            ),
            (
                'create_definite_value_proportion_features',
                FunctionTransformer(
                    partial(
                        definite_value_proportion_features_pipeline,
                        features_dictionary=prop_features_dict,
                        float_downcast_columns_list=float_downcast_columns_list
                    )
                )
            ),
            (
                'create_sum_prop_1_feature',
                FunctionTransformer(
                    from_is_zero_prop_1_create_sum_prop_1_feature_pipeline
                )
            ),

            (
                'from_pre_since_opened_create_pre_since_opened_sum_mean_repeated',
                FunctionTransformer(
                    pre_since_opened_sum_mean_repeated_pipeline
                )
            ),
            (
                'drop_temporary_source_columns',
                FunctionTransformer(
                    partial(
                        drop_columns_pipeline,
                        columns_list=drop_list
                    )
                )
            ),
            (
                'drop_duplicates_and_id',
                FunctionTransformer(
                    drop_duplicates_pipeline
                )
            ),
            (
                'classifier', classifier
            )

        ]
    )
    return main_pipe


def load_pipeline(
        path: str,
        logger: Optional[logging.Logger] = None
):
    """
    Загружает ранее сохранённый (обученный) пайплайн из файла.

    Проверяет наличие файла по указанному пути, выполняет загрузку объекта средствами pickle,
    при успешной загрузке выводит информационное сообщение в лог.

    Args:
        path (str): Путь к файлу с сохранённым пайплайном.
        logger (Optional[logging.Logger], default=None): Логгер для сообщений.
            Если None (по умолчанию), логирование этапов данной функции будет отключено.

    Returns:
        object: Загруженный pipeline.
    """
    try:
        with open(path, 'rb') as file:
            pipe = pickle.load(file)
        if logger is not None:
            logger.info(f'Pipeline loaded successfully from {path}')
        return pipe

    except FileNotFoundError:
        msg = (f'Pipeline file not found at {path}. '
               'Train the pipeline first (run with --mode train or without --mode flag).')
        if logger is not None:
            logger.error(msg)
        raise FileNotFoundError(msg)


def run_train_coordinator(
        pipeline_path: str,
        raw_data_path: str,
        temp_data_path: str,
        pre_features: List[str],
        num_parts_to_preprocess_at_once: int,
        pattern: str,
        target_path: str,
        train_size: float,
        seed_split_dataset: int,
        stratify_col: str,
        sample_frac: float,
        params_list: List[Dict],
        weights_list: List[float],
        threshold: float,
        cat_features: List[str],
        n_splits: int,
        seed: int,
        shuffle: bool,
        eval_metric: str,
        verbose: bool,
        drop_list_enc_paym_norm_summ_diff: List[str],
        mean_freq_source_list: List[str],
        drop_list_mean_value_frequency_feature: List[str],
        prop_features_dict: Dict[str, Any],
        float_downcast_columns_list: List[str],
        drop_list: List[str],
        classes_metric_list: List[str],
        cast_type_map: Optional[dict],
        search_file_ext: str,
        logger: Optional[logging.Logger] = None,
        mask: Optional[str] = None,

):
    """
    Запускает процесс обучения основного пайплайна на обучающих данных.

    Args:
        pipeline_path (str): Путь для сохранения обученного пайплайна.
        raw_data_path (str): Путь к исходной папке с "сырыми" parquet-данными для обучения.
        temp_data_path (str): Путь к папке для временного сохранения обработанных чанков данных.
        pre_features (List[str]): Список колонок исходных признаков, которые нужно оставить при загрузке данных.
        num_parts_to_preprocess_at_once (int): Сколько партиций данных обрабатывать за один проход.
        pattern (str): Маска расширения для поиска файлов. Отличается от search_file_ext даже
            при одном и том же расширении.
        target_path (str): Путь к CSV-файлу с целевой переменной (таргетом).
        train_size (float): Доля обучающей выборки (от 0 до 1).
        seed_split_dataset (int): Seed для разбиения на train/test (гарантирует воспроизводимость).
        stratify_col (str): Название колонки, по которой производится стратифицированное разбиение train/test.
        sample_frac (float): Доля строк исходных данных, используемая для вычисления медиан в SampleMedianImputer.
        params_list (List[Dict]): Список словарей с гиперпараметрами для каждой модели ансамбля CatBoost.
        weights_list (List[float]): Список весов для взвешивания предсказаний ансамбля моделей.
        threshold (float): Порог для жёсткой классификации.
        cat_features (List[str]): Список названий категориальных признаков.
        n_splits (int): Количество фолдов для ансамблирования моделей.
        seed (int): Seed для инициализации ансамблевого классификатора.
        shuffle (bool): Флаг перемешивания данных при разбиении на фолды.
        eval_metric (str): Режим расчёта метрик после обучения.
        verbose (bool): Включить прогресс-бары.
        drop_list_enc_paym_norm_summ_diff: List[str]: Список колонок на удаление в функции
            enc_paym_norm_group_sum_diff_pipeline.
        mean_freq_source_list (List[str]): Список признаков для расчёта средних частот значений в функции
            mean_value_frequency_feature_pipeline.
        drop_list_mean_value_frequency_feature: List[str]: Список колонок на удаление в функции
            mean_value_frequency_feature_pipeline.
        prop_features_dict (Dict[str, Any]): Словарь, определяющий признаки и значения для создания
            пропорциональных фичей в функции definite_value_proportion_features_pipeline.
        float_downcast_columns_list: List[str]: Список колонок  тип которых можно
            безопасно понизить с float64 до float32 без потери информативности
            из-за округления значений.
        drop_list (List[str]): Список признаков для удаления и очистки датасета на последнем этапе пайплайна
        classes_metric_list: (List[str]) Список метрик требующих метки классов для расчета.
            Используется в pred_and_metrics_compatible.
        cast_type_map : Словарь для приведения типов колонок {имя_колонки: тип},
            где тип — строка для приведения типа (например, 'int8', 'float32', 'category').
            Если None, типы не приводятся.
        search_file_ext (str): Расширение файлов для поиска (например, ".csv", ".pq").
            Отличается от pattern даже при одном и том же расширении.
        logger (Optional[logging.Logger], default=None): Логгер для сообщений.
            Если None (по умолчанию), логирование этапов данной функции будет отключено.
        mask (Optional[str], optional): Маска для выбора файлов в папке (например, 'train').
            Если указана, выбираются только файлы, имя которых начинается с mask;
            если None — выбираются все файлы.

    Последовательность действий:
        - Загружает исходный датасет с помощью функции load_dataset.
        - Загружает целевые значения и разделяет датасет на обучающую и тестовую выборки
          с помощью функции split_dataset_by_target.
        - Собирает и обучает полный pipeline (с препроцессингом, feature engineering и классификатором)
          на тренировочной части данных.
        - Сериализует обученный пайплайн в файл, путь к которому задаётся аргументом pipeline_path.
        - Протоколирует все ключевые этапы в логах.
        - При включении флгов --eval-metrics acc/auc обрабатывает тестовый набор данных,
             делает предикт, считает и логирует выбранную метрику.

    Returns:
        None

    Side-effect:
        сохранённый файл обученного пайплайна и логи выполнения.
    """
    if logger is not None:
        logger.info('Train mode started')

    # Получаем количество файлов в папке с данными
    files_count = check_data_folder_and_count_files(raw_data_path, pattern)[1]

    if logger is not None:
        logger.info('Loading raw dataset')
    # Загружаем датасет
    raw_data = load_dataset(
        path_to_dataset=raw_data_path,
        num_parts_to_preprocess_at_once=num_parts_to_preprocess_at_once,
        num_parts_total=files_count,
        save_to_path=temp_data_path,
        verbose=verbose,
        columns=pre_features,
        cast_type_map=cast_type_map,
        mask=mask,
        search_file_ext=search_file_ext
    )

    if logger is not None:
        logger.info('Splitting dataset into train and test sets')

    # Загружаем таргет
    # Делим датасет и таргет на train/test
    train_test_dict = split_dataset_by_target(
        dataset=raw_data,
        path_to_target=target_path,
        train_size=train_size,
        random_state=seed_split_dataset,
        stratify_col=stratify_col
    )

    # После разделения исходного датафрейма удаляем его для освобождения RAM
    del raw_data
    # Вызываем сборщика мусора
    gc.collect()

    if logger is not None:
        logger.info('Fitting the main pipeline')

    # Обучаем пайплайн
    pipe = main_pipeline(
        sample_frac=sample_frac,
        params_list=params_list,
        weights_list=weights_list,
        threshold=threshold,
        cat_features=cat_features,
        n_splits=n_splits,
        seed=seed,
        shuffle=shuffle,
        drop_list_enc_paym_norm_summ_diff=drop_list_enc_paym_norm_summ_diff,
        mean_freq_source_list=mean_freq_source_list,
        drop_list_mean_value_frequency_feature=drop_list_mean_value_frequency_feature,
        prop_features_dict=prop_features_dict,
        float_downcast_columns_list=float_downcast_columns_list,
        drop_list=drop_list,
        cast_type_map=cast_type_map,
        logger=logger
    ).fit(
        train_test_dict['X_train'],
        train_test_dict['y_train']
    )

    if logger is not None:
        logger.info(f'Saving trained pipeline to: {pipeline_path}')

    # Сохраним обученный пайплайн в pickle файл
    with open(pipeline_path, 'wb') as file:
        pickle.dump(pipe, file)

    # Считаем и логируем метрики
    # при переданном --eval-metrics acc/auc
    compute_and_log_metrics(
        eval_metric=eval_metric,
        pipe=pipe,
        train_test_dict=train_test_dict,
        classes_metric_list=classes_metric_list
    )
    if logger is not None:
        logger.info('Train mode completed successfully')


def run_test_coordinator(
        pipeline_path: str,
        raw_data_path: str,
        temp_data_path: str,
        pre_features: List[str],
        num_parts_to_preprocess_at_once: int,
        pattern: str,
        target_path: str,
        train_size: float,
        seed_split_dataset: int,
        stratify_col: str,
        test_predict_path: str,
        predict_file_extension: str,
        output: str,
        eval_metrics: str,
        classes_metric_list: List[str],
        verbose: bool,
        cast_type_map: Optional[dict],
        search_file_ext: str,
        logger: Optional[logging.Logger] = None,
        mask: Optional[str] = None
):
    """
    Выполняет тестирование обученного пайплайна на тестовой выборке.

    Args:
        pipeline_path (str): Путь к сериализованному sklearn pipeline.
        raw_data_path (str): Путь к директории с "сырыми" parquet-данными для тестирования.
        temp_data_path (str): Директория для временного хранения обработанных файлов.
        pre_features (List[str]): Список названий колонок, которые нужно загрузить из данных.
        num_parts_to_preprocess_at_once (int): Сколько партиций данных обрабатывать за один проход.
        pattern (str): Маска расширения для поиска файлов. Отличается от search_file_ext даже
            при одном и том же расширении.
        target_path (str): Путь к CSV-файлу с целевой переменной.
        train_size (float): Доля обучающей выборки (от 0 до 1 при разбиении train/test).
        seed_split_dataset (int): Seed для разделения на train/test (гарантирует воспроизводимость).
        stratify_col (str): Имя колонки, по которой выполняется стратификация при разделении.
        test_predict_path (str): Директория для сохранения предсказаний на тестовых данных.
        predict_file_extension: (str) Тип расширения файлов предиктов для функции make_file_path.
        output (str): Режим вывода предсказаний — 'proba' (вероятности классов)
            или 'predict' (жёсткая классификация).
        eval_metrics (str): Режим расчёта метрик на тестовой выборке ('off', 'auc', 'acc').
        classes_metric_list: (List[str]) Список метрик требующих метки классов для расчета.
            Используется в pred_and_metrics_compatible.
        verbose (bool): Включить  прогресс-бары.
        cast_type_map : Словарь для приведения типов колонок {имя_колонки: тип}.
            Если None, типы не приводятся.
        search_file_ext (str): Расширение файлов для поиска (например, ".csv", ".pq").
            Отличается от pattern даже при одном и том же расширении.
        logger (Optional[logging.Logger], default=None): Логгер для сообщений.
            Если None (по умолчанию), логирование этапов данной функции будет отключено.
        mask (Optional[str], optional): Маска для выбора файлов в папке (например, 'train').
            Если указана, выбираются только файлы, имя которых начинается с mask;
            если None — выбираются все файлы.

    Функция производит следующие этапы:
    - Загружает ранее обученный пайплайн (модель с этапами препроцессинга и feature engineering).
    - Загружает исходный датасет и разделяет его на обучающую и тестовую части.
    - Получает предсказания (вероятности классов либо метки классов) на тестовой подвыборке
      в зависимости от режима ('proba' или 'predict'), заданного через аргумент args.output.
    - Сохраняет полученные предсказания в соответствующий файл
          типа predict_raw_*.csv или proba_raw_*.csv.
    - При включении флгов --eval-metrics acc/auc  выводит в логи выбранную метрику.
    - Протоколирует каждый ключевой этап с помощью логера.

    Исключения:
    - Возникает и логируется ошибка, если обученный пайплайн не найден или не может быть загружен.

    Returns:
        None

    Side effect:
        файлы с предсказаниями и логи.

    Примечание:
    Для запуска функции необходимо наличие ранее обученного пайплайна
    (обратите внимание на режим обучения --mode train).
    """

    if logger is not None:
        logger.info('Test_coordinator started')

    if logger is not None:
        logger.info('Loading  the pipeline')

    # Пробуем загрузить обученный пайплайн,
    # если его нет то скрипт остановится с ошибкой.
    pipe = load_pipeline(pipeline_path)

    # Получаем количество файлов в папке с данными
    files_count = check_data_folder_and_count_files(raw_data_path, pattern)[1]

    if logger is not None:
        logger.info('Loading raw dataset')

    # Загружаем датасет
    raw_data = load_dataset(
        path_to_dataset=raw_data_path,
        num_parts_to_preprocess_at_once=num_parts_to_preprocess_at_once,
        num_parts_total=files_count,
        save_to_path=temp_data_path,
        verbose=verbose,
        columns=pre_features,
        cast_type_map=cast_type_map,
        mask=mask,
        search_file_ext=search_file_ext
    )

    if logger is not None:
        logger.info('Splitting dataset into train and test sets')

    # Загружаем таргет
    # Делим датасет и таргет на train/test
    train_test_dict = split_dataset_by_target(
        dataset=raw_data,
        path_to_target=target_path,
        train_size=train_size,
        random_state=seed_split_dataset,
        stratify_col=stratify_col
    )

    # Используем dispatch mapping для выбора жесткой или мягкой классификации
    # Создадим словарь режимов вывода
    output_handlers = {
        'proba': pipe.predict_proba,
        'predict': pipe.predict
    }
    # Получаем значение из парсера и вызываем соответствующий метод предикта
    handler = output_handlers.get(output)

    if logger is not None:
        logger.info(
            f'Getting {"probabilities" if output == "proba" else "classes"} for X_test data'
        )

    predictions = handler(
        train_test_dict['X_test']
    )
    # Cчитаем метрику по соответстывующему флагу.
    # Если полученный выше тип предикта соответствует метрике,
    # то используется он, иначе тестовая часть датасета снова обрабатывается
    # пайплайном для получения нужного типа предикта.
    compute_and_log_metrics(
        eval_metric=eval_metrics,
        pipe=pipe,
        train_test_dict=train_test_dict,
        classes_metric_list=classes_metric_list,
        y_pred=predictions
    )

    # БЛОК СОХРАНЕНИЯ ПРЕДИКТА

    # Меняем тип данных при мягкой классификации,
    # для понижения размера файла с предиктом
    if output == "proba":
        predictions = predictions.astype(np.float32)

    # Создаём имя файла предикта
    predict_file_name = make_file_path(
        output_type=output,
        data_path=raw_data_path,
        output_dir=test_predict_path,
        ext=predict_file_extension
    )
    if logger is not None:
        logger.info(
            f'Saving {"probabilities" if output == "proba" else "classes"}\n'
            f'to {predict_file_name}'
        )
    # Получаем id set для сохранения с предиктом
    # Используем drop_duplicates так как X_test это датасет до агрегаций в пайплайне
    ids = train_test_dict['X_test']['id'].drop_duplicates().values

    # Сохраненяем предикты в.csv
    save_predictions_with_id(
        output_type=output,
        ids=ids,
        predictions=predictions,
        output_path=predict_file_name
    )

    if logger is not None:
        logger.info('Test mode completed successfully')


def run_inference_coordinator(
        pipeline_path: str,
        data_path: str,
        max_files: int,
        temp_data_path: str,
        pre_features: List[str],
        num_parts_to_preprocess_at_once: int,
        pattern: str,
        predict_file_extension: str,
        output: str,
        output_dir: str,
        verbose: bool,
        cast_type_map: Optional[dict],
        search_file_ext: str,
        logger: Optional[logging.Logger] = None,
        mask: Optional[str] = None
):
    """
    Запускает режим инференса: загружает обученный пайплайн, подготавливает новые данные,
    вычисляет предсказания и сохраняет их в файл, поддерживая как вероятностный (proba),
    так и жёсткий (predict) режим вывода.

    Args:
        pipeline_path (str): Путь к сериализованному sklearn pipeline.
        data_path (str): Путь к директории с новыми данными.
        max_files (int, optional): Количество скачиваемых из папки файлов.
            Если не задано то качаются все файлы из папки.
        temp_data_path (str): Директория для временного хранения обработанных частей данных.
        pre_features (List[str]): Список колонок, которые нужно оставить при загрузке датасета.
        num_parts_to_preprocess_at_once (int): Сколько партиций данных обрабатывать за один проход.
        pattern (str): Маска расширения для поиска файлов. Отличается от search_file_ext даже
            при одном и том же расширении.
        predict_file_extension: (str) Тип расширения файлов предиктов для функции make_file_path
        output (str): Режим вывода предсказаний: 'proba' (вероятности классов) или 'predict' (метки классов).
        output_dir (str): Директория для сохранения итогового файла с предсказаниями.
        verbose (bool): Включить расширенный режим логирования и прогресс-бары.
        cast_type_map : Словарь для приведения типов колонок {имя_колонки: тип}.
            Если None, типы не приводятся.
        search_file_ext (str): Расширение файлов для поиска (например, ".csv", ".pq").
            Отличается от pattern даже при одном и том же расширении.
        logger (Optional[logging.Logger], default=None): Логгер для сообщений.
            Если None (по умолчанию), логирование этапов данной функции будет отключено.
        mask (Optional[str], optional): Маска для выбора файлов в папке (например, 'train').
            Если указана, выбираются только файлы, имя которых начинается с mask;
            если None — выбираются все файлы.

    Returns:
        None

    Side Effects:
        - Сохраняет файл с предсказаниями и колонкой id в директорию output_dir.
        - Записывает этапы вычислений в лог.

    Примечание:
    Для запуска функции необходимо наличие ранее обученного пайплайна
    (обратите внимание на режим обучения --mode train).
    """

    if logger is not None:
        logger.info('Inference mode started')

    if logger is not None:
        logger.info('Loading  the pipeline')

    # Пробуем загрузить обученный пайплайн,
    # если его нет то скрипт остановится с ошибкой.
    pipe = load_pipeline(pipeline_path)

    # Получаем количество файлов в папке с данными
    real_files_count = check_data_folder_and_count_files(data_path, pattern)[1]

    # Если количество файлов не задано через парсер,
    # то выбираем все файлы из паки.
    # Если задано, то выбираем минимум из заданного или реального количества файлов.
    if max_files is None:
        files_count = real_files_count
    else:
        files_count = min(real_files_count, max_files)

    if logger is not None:
        logger.info(
            f'Processing {files_count} files (max_files={max_files}, available={real_files_count})'
        )

    if logger is not None:
        logger.info(f'Loading dataset from : {data_path}')

    # Загружаем датасет
    data = load_dataset(
        path_to_dataset=data_path,
        num_parts_to_preprocess_at_once=num_parts_to_preprocess_at_once,
        num_parts_total=files_count,
        save_to_path=temp_data_path,
        verbose=verbose,
        columns=pre_features,
        cast_type_map=cast_type_map,
        mask=mask,
        search_file_ext=search_file_ext
    )
    # Используем dispatch mapping для выбора жесткой или мягкой классификации
    # Создадим словарь режимов вывода
    output_handlers = {
        'proba': pipe.predict_proba,
        'predict': pipe.predict
    }
    # Получаем значение из парсера и вызываем соответствующий метод предикта
    handler = output_handlers.get(output)

    if logger is not None:
        logger.info(
            f'Getting {"probabilities" if output == "proba" else "classes"} for {data_path}'
        )

    predictions = handler(data)

    # БЛОК СОХРАНЕНИЯ ПРЕДИКТА

    # Меняем тип данных при мягкой классификации,
    # для понижения размера файла с предиктом
    if output == "proba":
        predictions = predictions.astype(np.float32)

    # Получаем id set для сохранения с предиктом
    # Используем drop_duplicates так как X_test это датасет до агрегаций в пайплайне
    ids = data['id'].drop_duplicates().values

    # Создаём имя файла предикта
    predict_file_name = make_file_path(
        output,
        data_path,
        output_dir,
        ext=predict_file_extension
    )

    if logger is not None:
        logger.info(
            f'Saving {"probabilities" if output == "proba" else "classes"}\n'
            f'to {predict_file_name}'
        )

    # Сохраненяем предикты в .csv
    save_predictions_with_id(
        output_type=output,
        ids=ids,
        predictions=predictions,
        output_path=predict_file_name
    )

    if logger is not None:
        logger.info('Inference mode completed successfully')


def run_transform_split_coordinator(
        pipeline_path: str,
        raw_data_path: str,
        temp_data_path: str,
        pre_features: List[str],
        num_parts_to_preprocess_at_once: int,
        pattern: str,
        target_path: str,
        train_size: float,
        seed_split_dataset: int,
        stratify_col: str,
        verbose: bool,
        cast_type_map: Optional[dict],
        output_dir: str,
        transform_subset: str,
        search_file_ext: str,
        save_file_ext: str,
        output_type: str = 'transformed',
        logger: Optional[logging.Logger] = None,
        mask: Optional[str] = None,

):
    """
    Выполняет трансформацию тренировочной или тестовой выборки
    и сохраняет результат.

    Args:
        pipeline_path (str): Путь к сериализованному sklearn pipeline.
        raw_data_path (str): Путь к директории с "сырыми" parquet-данными.
        temp_data_path (str): Директория для временного хранения обработанных файлов.
        pre_features (List[str]): Список названий колонок, которые нужно загрузить из данных.
        num_parts_to_preprocess_at_once (int): Сколько партиций данных обрабатывать за один проход.
        pattern (str): Маска расширения для поиска файлов. Отличается от search_file_ext даже
            при одном и том же расширении.
        target_path (str): Путь к CSV-файлу с целевой переменной.
        train_size (float): Доля обучающей выборки (от 0 до 1 при разбиении train/test).
        seed_split_dataset (int): Seed для разделения на train/test (гарантирует воспроизводимость).
        stratify_col (str): Имя колонки, по которой выполняется стратификация при разделении.
        verbose (bool): Включить  прогресс-бары.
        cast_type_map : Словарь для приведения типов колонок {имя_колонки: тип}.
            Если None, типы не приводятся.
        output_dir (str): Директория для сохранения трансформированного файла.
        transform_subset (str): Выбор train/test подвыборки для трансформации.
        search_file_ext (str): Расширение файлов для поиска (например, ".csv", ".pq").
            Отличается от pattern даже при одном и том же расширении.
        save_file_ext (str): Расширение файла для сохранения результата (например, ".csv", ".pq").
        output_type (str, optional): Начало имени трансформированных файлов при сохранении.
            По умолчанию 'transformed'.
        logger (Optional[logging.Logger], default=None): Логгер для сообщений.
            Если None (по умолчанию), логирование этапов данной функции будет отключено.
        mask (Optional[str], optional): Маска для выбора файлов в папке (например, 'train').
            Если указана, выбираются только файлы, имя которых начинается с mask;
            если None — выбираются все файлы.

    Функция производит следующие этапы:
    - Загружает ранее обученный пайплайн (модель с этапами препроцессинга и feature engineering).
    - Загружает исходный датасет и разделяет его на обучающую и тестовую части.
    - Трансформирует train/test набор и сохраняет результат в csv файл.
    - Протоколирует каждый ключевой этап с помощью логера.

    Исключения:
    - Возникает и логируется ошибка, если обученный пайплайн не найден или не может быть загружен.

    Returns:
        None

    Side effect:
        сохраняет трансформированный файл.

    Примечание:
    Для запуска функции необходимо наличие ранее обученного пайплайна
    (обратите внимание на режим обучения --mode train).
    """
    if logger is not None:
        logger.info('Transform_split mode started')

    if logger is not None:
        logger.info(f'Transform subset selected: {transform_subset}')

    if logger is not None:
        logger.info('Loading  the pipeline')

    # Пробуем загрузить обученный пайплайн,
    # если его нет то скрипт остановится с ошибкой.
    pipe = load_pipeline(pipeline_path)

    # Получаем количество файлов в папке с данными
    files_count = check_data_folder_and_count_files(raw_data_path, pattern)[1]

    # Загружаем датасет
    if logger is not None:
        logger.info(f'Loading dataset from : {raw_data_path}')

    raw_data = load_dataset(
        path_to_dataset=raw_data_path,
        num_parts_to_preprocess_at_once=num_parts_to_preprocess_at_once,
        num_parts_total=files_count,
        save_to_path=temp_data_path,
        verbose=verbose,
        columns=pre_features,
        cast_type_map=cast_type_map,
        mask=mask,
        search_file_ext=search_file_ext
    )

    if logger is not None:
        logger.info('Splitting dataset into train and test sets')

    # Загружаем таргет
    # Делим датасет и таргет на train/test
    train_test_dict = split_dataset_by_target(
        dataset=raw_data,
        path_to_target=target_path,
        train_size=train_size,
        random_state=seed_split_dataset,
        stratify_col=stratify_col
    )

    # После разделения исходного датафрейма удаляем его
    # для освобождения RAM
    del raw_data
    # Вызываем сборщика мусора
    gc.collect()

    # Выбираем подвыборку для трансформации
    if transform_subset == 'train':
        data_to_transform = train_test_dict['X_train']
        subset_name = 'train'
    else:
        data_to_transform = train_test_dict['X_test']
        subset_name = 'test'

    # Трансформируем данные
    transformed_data = pipe.transform(data_to_transform)

    # Создаём путь для сохранения
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    file_name = f"{output_dir}/{output_type}_{subset_name}_{timestamp}.{save_file_ext}"

    if logger is not None:
        logger.info(
            f"Saving transformed {subset_name} subset to: {file_name}")

    # Сохраняем результат
    transformed_data.to_csv(file_name, index=False)

    if logger is not None:
        logger.info('Transform_split mode completed successfully')


def run_transform_data_coordinator(
        pipeline_path: str,
        data_path: str,
        max_files: int,
        temp_data_path: str,
        pre_features: List[str],
        num_parts_to_preprocess_at_once: int,
        pattern: str,
        output_dir: str,
        verbose: bool,
        cast_type_map: Optional[dict],
        search_file_ext: str,
        save_file_ext: str,
        output_type: str = 'transformed',
        logger: Optional[logging.Logger] = None,
        mask: Optional[str] = None
):
    """
    Выполняет трансформацию  данных и сохраняет результат.

    Args:
        pipeline_path (str): Путь к сериализованному sklearn pipeline.
        data_path (str): Путь к директории с новыми входными данными.
        max_files (int, optional): Количество скачиваемых из папки файлов.
            Если не задано то качаются все файлы из папки.
        temp_data_path (str): Директория для временного хранения обработанных частей данных.
        pre_features (List[str]): Список колонок, которые нужно оставить при загрузке датасета.
        num_parts_to_preprocess_at_once (int): Сколько партиций данных обрабатывать за один проход.
        pattern (str): Маска расширения для поиска файлов. Отличается от search_file_ext даже
            при одном и том же расширении.
        output_dir (str): Директория для сохранения трансформированного файла.
        verbose (bool): Включить прогресс-бары.
        cast_type_map : Словарь для приведения типов колонок {имя_колонки: тип}.
            Если None, типы не приводятся.
        search_file_ext (str): Расширение файлов для поиска (например, ".csv", ".pq").
            Отличается от pattern даже при одном и том же расширении.
        save_file_ext (str): Расширение файла для сохранения результата (например, ".csv", ".pq").
        output_type (str, optional): Начало имени трансформированных файлов при сохранении.
            По умолчанию 'transformed'.
        logger (Optional[logging.Logger], default=None): Логгер для сообщений.
            Если None (по умолчанию), логирование этапов данной функции будет отключено.
        mask (Optional[str], optional): Маска для выбора файлов в папке (например, 'train').
            Если указана, выбираются только файлы, имя которых начинается с mask;
            если None — выбираются все файлы.

    Returns:
        None

    Side Effects:
        - Сохраняет файл с предсказаниями и колонкой id в директорию output_dir.
        - Записывает этапы вычислений в лог.

    Примечание:
    Для запуска функции необходимо наличие ранее обученного пайплайна
    (обратите внимание на режим обучения --mode train).
    """
    if logger is not None:
        logger.info('Transform_data mode started')

    if logger is not None:
        logger.info('Loading  the pipeline')

    # Пробуем загрузить обученный пайплайн,
    # если его нет то скрипт остановится с ошибкой.
    pipe = load_pipeline(pipeline_path)

    # Получаем количество файлов в папке с данными
    real_files_count = check_data_folder_and_count_files(data_path, pattern)[1]

    # Если количество файлов не задано через парсер,
    # то выбираем все файлы из паки.
    # Если задано, то выбираем минимум из заданного или реального количества файлов.
    if max_files is None:
        files_count = real_files_count
    else:
        files_count = min(real_files_count, max_files)

    if logger is not None:
        logger.info(
            f'Processing {files_count} files (max_files={max_files}, available={real_files_count})'
        )

    if logger is not None:
        logger.info(f'Loading dataset from : {data_path}')

    # Загружаем датасет
    data = load_dataset(
        path_to_dataset=data_path,
        num_parts_to_preprocess_at_once=num_parts_to_preprocess_at_once,
        num_parts_total=files_count,
        save_to_path=temp_data_path,
        verbose=verbose,
        columns=pre_features,
        cast_type_map=cast_type_map,
        mask=mask,
        search_file_ext=search_file_ext
    )

    # Трансформируем даные
    transformed_data = pipe.transform(data)

    # Создаём имя файла
    file_name = make_file_path(
        output_type=output_type,
        data_path=data_path,
        output_dir=output_dir,
        ext=save_file_ext
    )

    if logger is not None:
        logger.info(
            f"Saving transformed data to: {file_name}")

    # Сохраняем результат
    transformed_data.to_csv(file_name, index=False)

    if logger is not None:
        logger.info('Transform_data mode completed successfully')



if __name__ == "__main__":
    # Парсим аргументы из командной строки
    args = parser.parse_args()

    # Получаем логер из импортированной функции.
    # Настраиваем логирование на основе аргумента.
    logger = setup_logging(args.log_level)

    # Синхронизация verbose с --log-level (True если 'info' или 'debug', False если 'off')
    # lля вывода  баров загрузки в функции load_dataset
    verbose = args.log_level in ['info', 'debug']

    if logger is not None:
        logger.info('Pipeline started')

    # Используем dispatch mapping
    # Создадим словарь режимов пайплайна.
    # Координаторы будем вызывать через lambda и передавать параметры
    # из конфига и парсера(один параметр вручную).
    # ВАЖНО: именно здесь происходит первичная передача всех параметров в алгоритм;
    # далее параметры прокидываются по функциям и классам явно,
    # без повторного определения или извлечения из внешних источников.
    mode_handlers = {
        'train': lambda: run_train_coordinator(
            pipeline_path=PIPELINE_PATH,
            raw_data_path=RAW_DATA_PATH,
            temp_data_path=TEMP_DATA_PATH,
            pre_features=PRE_FEATURES,
            num_parts_to_preprocess_at_once=1,
            pattern=PARQUET_FILE_PATTERN,
            target_path=TARGET_PATH,
            train_size=TRAIN_SIZE,
            seed_split_dataset=SEED_SPLIT_DATASET,
            stratify_col=STRATIFY_COL,
            sample_frac=SAMPLE_FRAC,
            params_list=PARAMS_LIST,
            weights_list=WEIGHTS_LIST,
            threshold=THRESHOLD,
            cat_features=CAT_FEATURES,
            n_splits=N_SPLITS,
            seed=SEED,
            shuffle=SHUFFLE,
            eval_metric=args.eval_metrics,
            verbose=verbose,
            drop_list_enc_paym_norm_summ_diff=DROP_LIST_ENC_PAYM_NORM_GROUP_SUMM_DIFF,
            mean_freq_source_list=MEAN_FREQ_SOURCE_LIST,
            drop_list_mean_value_frequency_feature=DROP_LIST_MEAN_VALUE_FREQUENCY_FEATURE,
            prop_features_dict=PROP_FEATURES_DICT,
            float_downcast_columns_list=FLOAT_DOWNCAST_COLUMNS_LIST,
            drop_list=DROP_LIST,
            classes_metric_list=CLASSES_METRIC_LIST,
            search_file_ext=SEARCH_FILE_EXTENSION,
            cast_type_map=CAST_TYPE_MAP,
            logger=logger
        ),
        'test': lambda: run_test_coordinator(
            pipeline_path=PIPELINE_PATH,
            raw_data_path=RAW_DATA_PATH,
            temp_data_path=TEMP_DATA_PATH,
            pre_features=PRE_FEATURES,
            num_parts_to_preprocess_at_once=1,
            pattern=PARQUET_FILE_PATTERN,
            target_path=TARGET_PATH,
            train_size=TRAIN_SIZE,
            seed_split_dataset=SEED_SPLIT_DATASET,
            stratify_col=STRATIFY_COL,
            test_predict_path=TEST_PREDICT_PATH,
            predict_file_extension=SAVE_FILE_EXTENSION,
            output=args.output,
            eval_metrics=args.eval_metrics,
            classes_metric_list=CLASSES_METRIC_LIST,
            verbose=verbose,
            search_file_ext=SEARCH_FILE_EXTENSION,
            cast_type_map=CAST_TYPE_MAP,
            logger=logger,
        ),
        'inference': lambda: run_inference_coordinator(
            pipeline_path=PIPELINE_PATH,
            data_path=args.data_path,
            max_files=args.max_files,
            temp_data_path=TEMP_DATA_PATH,
            pre_features=PRE_FEATURES,
            num_parts_to_preprocess_at_once=1,
            pattern=PARQUET_FILE_PATTERN,
            predict_file_extension=SAVE_FILE_EXTENSION,
            output=args.output,
            output_dir=args.output_dir,
            verbose=verbose,
            search_file_ext=SEARCH_FILE_EXTENSION,
            cast_type_map=CAST_TYPE_MAP,
            logger=logger
        ),
        'transform_split': lambda: run_transform_split_coordinator(
            pipeline_path=PIPELINE_PATH,
            raw_data_path=RAW_DATA_PATH,
            temp_data_path=TEMP_DATA_PATH,
            pre_features=PRE_FEATURES,
            num_parts_to_preprocess_at_once=1,
            pattern=PARQUET_FILE_PATTERN,
            target_path=TARGET_PATH,
            train_size=TRAIN_SIZE,
            seed_split_dataset=SEED_SPLIT_DATASET,
            stratify_col=STRATIFY_COL,
            verbose=verbose,
            cast_type_map=CAST_TYPE_MAP,
            output_dir=TRANSFORM_DATA_PATH,
            transform_subset=args.transform_subset,
            search_file_ext=SEARCH_FILE_EXTENSION,
            save_file_ext=SAVE_FILE_EXTENSION,
            logger=logger
        ),
        'transform_data': lambda: run_transform_data_coordinator(
            pipeline_path=PIPELINE_PATH,
            data_path=args.data_path,
            max_files=args.max_files,
            temp_data_path=TEMP_DATA_PATH,
            pre_features=PRE_FEATURES,
            num_parts_to_preprocess_at_once=1,
            pattern=PARQUET_FILE_PATTERN,
            output_dir=TRANSFORM_DATA_PATH,
            verbose=verbose,
            cast_type_map=CAST_TYPE_MAP,
            logger=logger,
            search_file_ext=SEARCH_FILE_EXTENSION,
            save_file_ext=SAVE_FILE_EXTENSION
    )
    }
    # Получим значение из парсера и
    # запустим соответствующий режим пайплайна
    handler = mode_handlers.get(args.mode)
    handler()

    if logger is not None:
        logger.info("Pipeline completed successfully")