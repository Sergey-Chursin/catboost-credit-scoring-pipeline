import os

import argparse
import glob
import pickle
from typing import Optional, List, Dict
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
    TEST_PREDICT_PATH
)

from data_utils import (
    load_dataset,
    split_dataset_by_target,
    check_data_folder_and_count_files,
    make_file_path,
    save_predictions_with_id
)

from evaluate_metrics import compute_and_log_metrics


# Переключатель уровня логирования
from log_config import setup_logging

from preprocessing import (
    SampleMedianImputer,
    convert_all_to_numeric_pipeline,
    convert_all_to_int_pipeline,
    drop_duplicates_pipeline
)

from feature_engineering import (
    rn_max_feature_pipeline,
    enc_paym_transcoding_pipeline,
    definite_value_proportion_features_pipeline,
    from_is_zero_prop_1_create_sum_prop_1_feature_pipeline,
    mean_value_frequency_feature_pipeline,
    enc_paym_norm_group_sum_diff_pipeline,
    pre_since_opened_sum_mean_repeated_pipeline,
    drop_columns_drop_duplicates_pipeline
)
from classifier import CatBoostEnsembleClassifier

"""
Настраиваем парсер аргументов для CLI-запуска.
Это позволяет запускать скрипт с флагами:
--log-level info   для вывода логов;

--mode train        для загрузки тренировочного датасета, 
                    разделения его на train/test,
                    обучения пайплайна train,
                    сохранения пайплайна.               
--mode test         для загрузки тренировочного датасета, 
                    разделения его на train/test,
                    загрузки пайплайна,
                    получения и сохранения предикта.               
--mode inference    для загрузки тренировочного датасета,
                    получения и сохранения предикта.
                    Это имитирует получение предикта 
                    на новых данных. За неимением других
                    данных тестируем на всём тренировочном
                    датасете.
                  
--output proba      для режимов test/inference 
                    получение вероятностей классов.
--output predict    для режимов test/inference 
                    жесткая классификация.
                    
--data-path         путь к данным для inference
                    по умолчанию это путь
                    к тренировочному датасету.   
                    
--eval-metrics off нет вывода метрик на тестовой выборке 
--eval-metrics auc на тестовой выборки считается AUC SCORE
                     и выводится в логи.
--eval-metrics acc на тестовой выборки считается ACCURACY
                     и выводится в логи.   
                     
--output-dir str   путь сохранения предиктов на новых данных,
                   по умолчанию /../predictions/iference/
                                                              
Флаги можно ставить в любом порядке.
Выбор флага --output... в --mode train не вызовет ошибки,
сработает скрипт обучения. 
--help - Вывод help-сообщения
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
default - значение по умолчанию, этот флаг можно не вводить.
"""
# Режим логирования. По умолчанию логи отключены.
parser.add_argument(
    '--log-level',
    type=str,
    default='off',
    choices=['info', 'off'],
    help=(
        'Logging level: \n'
        'info - enable detailed logs\n'
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
    choices=['train', 'test', 'inference'],
    help=(
        'Execution mode:\n'
        'train - fit and save model\n'
        'test - validate on the test set\n'
        'inference - predict on new data\n'
        'Default: train\n'
        'Example: --mode test'
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
# Путь к новым данным для режима inference
# Реализован через os для кроссплатформенности
parser.add_argument(
    '--data-path',
    type=str,
    default=os.path.join('..', 'data', 'raw'),
    help=(
        'Path to data folder containing .pq (Parquet) files.\n'
        'This is used only for --mode inference to specify new data.\n'
        'In train/test modes, it is ignored — fixed paths from config are used instead.\n'
        'The script loads and concatenates all .pq files in the folder.\n'
        'Default: ../data/train/ (training data path).\n'
        'Example (for inference): --data-path /path/to/new_data/'
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


def main_pipeline(
        params_list: List[Dict],
        weights_list: List[float],
        sample_frac: float
):
    """
    Создаёт и возвращает основной Pipeline для обучения и предсказания.

    Pipeline включает стандартные этапы препроцессинга и feature engineering
     над табличными данными,
    а также кастомный классификатор на основе ансамбля моделей CatBoost.

    Args:
        params_list (List[Dict]): Список словарей с гиперпараметрами
            для каждой модели CatBoost в ансамбле.
        weights_list (List[float]): Список весов для агрегации предсказаний ансамбля.
        sample_frac (float): Доля строк исходных данных, используемая для
            вычисления медиан в SampleMedianImputer.

    Returns:
        sklearn.pipeline.Pipeline: Собранный pipeline, готовый для обучения (fit)
            или предсказания (predict/predict_proba).

    Пример использования:
        pipe = main_pipeline()
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
    """
    # Создаём SampleMedianImputer для заполнения пустых значений медианами
    imputer = SampleMedianImputer(sample_frac=sample_frac)

    # Создаём паплайн препроцессинга данных
    preprocessing_pipe = Pipeline([
        ('to_numeric', FunctionTransformer(convert_all_to_numeric_pipeline)),
        ('imputer', imputer),
        ('to_int', FunctionTransformer(convert_all_to_int_pipeline)),
        ('drop_duplicates', FunctionTransformer(drop_duplicates_pipeline))
    ])

    # Создаём основной пайплайн
    main_pipe = Pipeline(
        [
            (
                'preprocessing',
                preprocessing_pipe
            ),
            (
                'create_rn_max_feature',
                FunctionTransformer(rn_max_feature_pipeline)
            ),
            (
                'enc_paym_transcoding',
                FunctionTransformer(enc_paym_transcoding_pipeline)
            ),
            (
                'create_definite_value_proportion_features',
                FunctionTransformer(definite_value_proportion_features_pipeline)
            ),
            (
                'create_sum_prop_1_feature',
                FunctionTransformer(from_is_zero_prop_1_create_sum_prop_1_feature_pipeline)
            ),
            (
                'create_mean_value_frequency_feature',
                FunctionTransformer(mean_value_frequency_feature_pipeline)
            ),
            (
                'from_enc_paym_create_normalized_group_sum_features_then_diff_features',
                FunctionTransformer(enc_paym_norm_group_sum_diff_pipeline)
            ),
            (
                'from_pre_since_opened_create_pre_since_opened_sum_mean_repeated',
                FunctionTransformer(pre_since_opened_sum_mean_repeated_pipeline)
            ),
            (
                'drop_temporary_and_source_columns_drop_duplicates',
                FunctionTransformer(drop_columns_drop_duplicates_pipeline)
            ),
            (
                'classifier',
                CatBoostEnsembleClassifier(
                    params_list=params_list,
                    weights_list=weights_list,
                    logger=logger
                )
            )

        ]
    )

    return main_pipe

def load_pipeline(path: str):
    """
    Загружает обученный пайплайн с проверкой его
    существования.
    """
    try:
        with open(path, 'rb') as file:
            pipe = pickle.load(file)
        logger.info(f'Pipeline loaded successfully from {path}')
        return pipe

    except FileNotFoundError:
        msg = (f'Pipeline file not found at {path}. '
               'Train the pipeline first (run with --mode train or without --mode flag).')
        logger.error(msg)
        raise FileNotFoundError(msg)

def train_coordinator(
        pipeline_path: str,
        raw_data_path: str,
        temp_data_path: str,
        pre_features: List[str],
        num_parts_to_preprocess_at_once: int,
        target_path: str,
        train_size: float,
        seed_split_dataset: int,
        stratify_col: str,
        params_list: List[Dict],
        weights_list: List[str],
        sample_frac: float,
        eval_metric: str,
        verbose: bool
):
    """
    Запускает процесс обучения основного пайплайна на обучающих данных.

    Args:
        Все параметры по умолчанию берутся из config.py.

        pipeline_path (str): Путь для сохранения обученного пайплайна (модель + препроцессинг).
        raw_data_path (str): Путь к исходной папке с "сырыми" parquet-данными для обучения.
        temp_data_path (str): Директория для временного сохранения обработанных чанков данных.
        pre_features (List[str]): Список колонок исходных признаков, которые нужно оставить
            при загрузке данных.
        num_parts_to_preprocess_at_once (int): Сколько партиций данных обрабатывать за один проход.
        target_path (str): Путь к CSV-файлу с целевой переменной (таргетом).
        train_size (float): Доля обучающей выборки (значение от 0 до 1).
        seed_split_dataset (int): Seed для разбиения на train/test (гарантирует воспроизводимость).
        stratify_col (str): Название колонки, по которой производится стратифицированное разбиение train/test.
        params_list (List[Dict]): Список словарей с гиперпараметрами для каждой модели ансамбля CatBoost.
        weights_list (List[str]): Список весов для взвешивания предсказаний ансамбля моделей.
        sample_frac (float): Доля строк исходных данных, используемая для вычисления медиан в импутере.
        eval_metric (str): Режим расчёта метрик после обучения.
        verbose (bool): Включить расширенный режим логирования и прогресс-бары.

    Последовательность действий:
        - Загружает основной исходный датасет с помощью функции load_dataset.
        - Загружает целевые значения и разделяет датасет на обучающую и тестовую выборки
          с помощью функции split_dataset_by_target.
        - Собирает и обучает полный pipeline (с препроцессингом, feature engineering и классификатором)
          на тренировочной части данных.
        - Сериализует обученный пайплайн в файл, путь к которому задаётся аргументом path.
        - Протоколирует все ключевые этапы в логах.

    Возвращаемое значение:
        - Ничего не возвращает (side-effect: сохранённый файл обученного пайплайна и логи выполнения).

    Примечания:
        - Функция предназначена для запуска в режиме обучения
            (с флагом --mode train или без флага --mode вообще).
    """

    logger.info('Train mode started')

    # Получаем количество файлов в папке с данными
    files_count = check_data_folder_and_count_files(raw_data_path)[1]

    # Загружаем датасет
    logger.info('Loading raw dataset')
    raw_data = load_dataset(
        path_to_dataset=raw_data_path,
        num_parts_to_preprocess_at_once=num_parts_to_preprocess_at_once,
        num_parts_total=files_count,
        save_to_path=temp_data_path,
        verbose=verbose,
        columns=pre_features
    )

    # Загружаем таргет
    # Делим датасет и таргет на train/test
    logger.info('Splitting dataset into train and test sets')
    train_test_dict = split_dataset_by_target(
        dataset=raw_data,
        path_to_target=target_path,
        train_size=train_size,
        random_state=seed_split_dataset,
        stratify_col=stratify_col,
        verbose=verbose)

    # Обучаем пайплайн
    logger.info('Fitting the main pipeline')
    pipe = main_pipeline(
        params_list=params_list,
        weights_list=weights_list,
        sample_frac=sample_frac
    ).fit(
        train_test_dict['X_train'],
        train_test_dict['y_train']
    )

    # Сохраним обученный пайплайн в файл
    logger.info(f'Saving trained pipeline to: {pipeline_path}')
    with open(pipeline_path, 'wb') as file:
        pickle.dump(pipe, file)

    # Считаем и логируем метрики
    compute_and_log_metrics(
        eval_metric=eval_metric,
        pipe=pipe,
        train_test_dict=train_test_dict
    )

    logger.info('Train mode completed successfully')

def test_coordinator(
        pipeline_path: str,
        raw_data_path: str,
        temp_data_path: str,
        pre_features: List[str],
        num_parts_to_preprocess_at_once: int,
        target_path: str,
        train_size: float,
        seed_split_dataset: int,
        stratify_col: str,
        test_predict_path: str,
        output: str,
        eval_metrics: str,
        verbose: bool
):
    """
    Выполняет тестирование обученного пайплайна на тестовой выборке.

    Args:
        pipeline_path (str): Путь к сериализованному sklearn pipeline
            (модель + препроцессинг/feature engineering).
        raw_data_path (str): Путь к директории с "сырыми" parquet-данными для тестирования.
        temp_data_path (str): Директория для временного хранения обработанных файлов.
        pre_features (List[str]): Список названий колонок, которые нужно загрузить из данных.
        num_parts_to_preprocess_at_once (int): Сколько партиций данных обрабатывать за один проход.
        target_path (str): Путь к CSV-файлу с целевой переменной.
        train_size (float): Доля обучающей выборки (от 0 до 1 при разбиении train/test).
        seed_split_dataset (int): Seed для разделения на train/test (гарантирует воспроизводимость).
        stratify_col (str): Имя колонки, по которой выполняется стратификация при разделении.
        test_predict_path (str): Директория/файл для сохранения предсказаний на тестовых данных.
        output (str): Режим вывода предсказаний — 'proba' (вероятности классов)
            или 'predict' (жёсткая классификация).
        eval_metrics (str): Режим расчёта метрик на тестовой выборке ('off', 'auc', 'acc').
        verbose (bool): Включить расширенный режим логирования и прогресс-бары.

    Функция производит следующие этапы:
    - Загружает ранее обученный пайплайн (модель с этапами препроцессинга и feature engineering).
    - Загружает исходный датасет и разделяет его на обучающую и тестовую части.
    - Получает предсказания (вероятности классов либо метки классов) на тестовой подвыборке
      в зависимости от режима ('proba' или 'predict'), заданного через аргумент args.output.
    - Сохраняет полученные предсказания в соответствующий файл (test_proba.pkl или test_classes.pkl).
    - Протоколирует каждый ключевой этап с помощью логгера.

    Исключения:
    - Возникает и логируется ошибка, если обученный пайплайн не найден или не может быть загружен.

    Возвращаемое значение:
    - Ничего не возвращает (side effect: файлы с предсказаниями и логи).

    Примечание:
    Для запуска функции необходимо наличие ранее обученного пайплайна
    (обратите внимание на режим обучения --mode train).
    """
    logger.info('Test_coordinator started')
    """
    Пробуем загрузить обученный пайплайн,
    если его нет то скрипт остановится с ошибкой.
    """
    logger.info('Loading  the pipeline')
    pipe = load_pipeline(pipeline_path)

    # Получаем количество файлов в папке с данными
    files_count = check_data_folder_and_count_files(raw_data_path)[1]

    # Загружаем датасет
    logger.info('Loading raw dataset')
    raw_data = load_dataset(
        path_to_dataset=raw_data_path,
        num_parts_to_preprocess_at_once=num_parts_to_preprocess_at_once,
        num_parts_total=files_count,
        save_to_path=temp_data_path,
        verbose=verbose,
        columns = pre_features
    )

    # Загружаем таргет
    # Делим датасет и таргет на train/test
    logger.info('Splitting dataset into train and test sets')
    train_test_dict = split_dataset_by_target(
        dataset=raw_data,
        path_to_target=target_path,
        train_size=train_size,
        random_state=seed_split_dataset,
        stratify_col=stratify_col,
        verbose=verbose)

    # Используем dispatch mapping для выбора жесткой или мягкой классификации
    # Создадим словарь режимов вывода
    output_handlers = {
        'proba': pipe.predict_proba,
        'predict': pipe.predict
    }
    # Получаем значение из парсера и вызываем соответствующий метод предикта
    handler = output_handlers.get(output)
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
        ext="csv"
    )

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

    logger.info('Test mode completed successfully')

def inference_coordinator(
        pipeline_path: str,
        data_path: str,
        temp_data_path: str,
        pre_features: List[str],
        num_parts_to_preprocess_at_once: int,
        output: str,
        output_dir: str,
        verbose: bool,
):

    """
    Запускает режим инференса: загружает обученный пайплайн, подготавливает новые данные,
    вычисляет предсказания и сохраняет их в файл, поддерживая как вероятностный (proba),
    так и жёсткий (predict) режим вывода.

    Args:
        pipeline_path (str): Путь к сериализованному sklearn pipeline
            (модель + препроцессинг/feature engineering).
        data_path (str): Путь к директории с новыми входными данными для инференса.
        temp_data_path (str): Директория для временного хранения обработанных частей данных.
        pre_features (List[str]): Список колонок, которые нужно оставить при загрузке нового датасета.
        num_parts_to_preprocess_at_once (int): Сколько партиций данных обрабатывать за один проход.
        output (str): Режим вывода предсказаний: 'proba' (вероятности классов) или 'predict' (метки классов).
        output_dir (str): Директория для сохранения итогового файла с предсказаниями.
        verbose (bool): Включить расширенный режим логирования и прогресс-бары.

    Returns:
        None

    Side Effects:
        - Сохраняет файл с предсказаниями и колонкой id в директорию output_dir.
        - Записывает этапы вычислений в лог.
    """
    logger.info('Inference mode started')
    """
    Пробуем загрузить обученный пайплайн,
    если его нет то скрипт остановится с ошибкой.
    """
    logger.info('Loading  the pipeline')
    pipe = load_pipeline(pipeline_path)

    # Получаем количество файлов в папке с данными
    files_count = check_data_folder_and_count_files(data_path)[1]

    # Загружаем датасет
    logger.info(f'Loading dataset from : {data_path}')
    data = load_dataset(
        path_to_dataset = data_path,
        num_parts_to_preprocess_at_once = num_parts_to_preprocess_at_once,
        num_parts_total = files_count,
        save_to_path = temp_data_path,
        verbose = verbose,
        columns = pre_features
    )
    # Используем dispatch mapping для выбора жесткой или мягкой классификации
    # Создадим словарь режимов вывода
    output_handlers = {
        'proba': pipe.predict_proba,
        'predict': pipe.predict
    }
    # Получаем значение из парсера и вызываем соответствующий метод предикта
    handler = output_handlers.get(output)
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
        output_dir
    )
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

    logger.info('Inference mode completed successfully')

if __name__ == "__main__":
    # Парсим аргументы из командной строки
    args = parser.parse_args()
    """
    Получаем логер из импортированной функции.
    Настраиваем логирование на основе аргумента.
    'info' включит логи, 'off' — отключит.
    """
    logger = setup_logging(args.log_level)

    """
    Синхронизация verbose с --log-level (True если 'info', False если 'off')
    Для вывода логов и бара загрузки в функции prepare_transactions_dataset
    """
    verbose = args.log_level == 'info'

    logger.info('Pipeline started')

    # Используем dispatch mapping
    # Создадим словарь режимов пайплайна.
    # Координаторы будем вызывать через lambda и передавать параметры из конфига
    mode_handlers = {
        'train': lambda: train_coordinator(
            pipeline_path=PIPELINE_PATH,
            raw_data_path=RAW_DATA_PATH,
            temp_data_path=TEMP_DATA_PATH,
            pre_features=PRE_FEATURES,
            num_parts_to_preprocess_at_once=1,
            target_path=TARGET_PATH,
            train_size=TRAIN_SIZE,
            seed_split_dataset=SEED_SPLIT_DATASET,
            stratify_col=STRATIFY_COL,
            params_list = PARAMS_LIST,
            weights_list = WEIGHTS_LIST,
            sample_frac = SAMPLE_FRAC,
            eval_metric=args.eval_metrics,
            verbose=verbose

        ),
        'test': lambda: test_coordinator(
            pipeline_path = PIPELINE_PATH,
            raw_data_path = RAW_DATA_PATH,
            temp_data_path = TEMP_DATA_PATH,
            pre_features = PRE_FEATURES,
            num_parts_to_preprocess_at_once=1,
            target_path = TARGET_PATH,
            train_size = TRAIN_SIZE,
            seed_split_dataset = SEED_SPLIT_DATASET,
            stratify_col = STRATIFY_COL,
            test_predict_path = TEST_PREDICT_PATH,
            output = args.output,
            eval_metrics = args.eval_metrics,
            verbose=verbose
        ),
        'inference': lambda: inference_coordinator(
            pipeline_path = PIPELINE_PATH,
            data_path=args.data_path,
            temp_data_path = TEMP_DATA_PATH,
            pre_features = PRE_FEATURES,
            num_parts_to_preprocess_at_once = 1,
            output = args.output,
            output_dir = args.output_dir,
            verbose = verbose
        )
    }
    # Получим значение из парсера и
    # запустим соответствующий режим пайплайна
    handler = mode_handlers.get(args.mode)
    handler()

    # Удалить
    logger.info("Pipeline completed")