import os

# Модуль с константами, параметрами, путями, списками и словарями для пайплайна.

"""
Получаем абсолютный  путь до корня проекта.
os.path.dirname(__file__) - даёт абсолютный путь до директории текущего файла (src)
os.path.join(..., '..')) - даёт путь до директории на уровень выше (root)
os.path.abspath(...) - даёт абсолютный путь до root
"""
PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Паттерн расширения для функции check_data_folder_and_count_files

PARQUET_FILE_PATTERN: str = "*.pq"

# Расширение файлов для функции load_data_chunks
SEARCH_FILE_EXTENSION: str = ".pq"

# Константы для функции load_dataset
# Путь к директории с исходными данными
RAW_DATA_PATH: str = os.path.join(PROJECT_ROOT, "data", "raw")
# Путь к директории с техническими данными
TEMP_DATA_PATH: str = os.path.join(PROJECT_ROOT, "data", "temp")

# Тип расширения для функции make_file_path для сохранения предиктов
# в run_test_coordinator и run_inference_coordinator
# и транформированных данных в run_transform_data_coordinator
SAVE_FILE_EXTENSION: str = "csv"

# Константы для функции split_dataset_by_target
# Путь к таргет датасету
TARGET_PATH: str = os.path.join(PROJECT_ROOT, "data", "target", "train_target.csv")
# Доля тренировочной выборки
TRAIN_SIZE: float = 0.8
# Зерно рандома для разделения
SEED_SPLIT_DATASET: int = 0
# Колонка для стратификации
STRATIFY_COL: str = "flag"

# Путь сохранения предиктов на новых данных в inference_coordinator
INFERENCE_OUTPUT_DIR: str = os.path.join(PROJECT_ROOT, "predictions", "inference")

# Доля датасета для расчета медиан в SampleMedianImputer
SAMPLE_FRAC: float = 0.1
# Путь сохранения обученного пайплайна
PIPELINE_PATH: str = os.path.join(PROJECT_ROOT, "models", "main_pipeline.pkl")

DEMO_PIPELINE_PATH: str = os.path.join(PROJECT_ROOT, "models", "demo_pipeline.pkl")

# Путь сохранения предиктов test_coordinator
TEST_PREDICT_PATH: str = os.path.join(PROJECT_ROOT, "predictions")

# Путь сохранения трансформированных данных
TRANSFORM_DATA_PATH: str = os.path.join(PROJECT_ROOT, "data", "processed")

# Паттерн пути до сохранённого предикта вероятностей на тестовом наборе
PROBA_TEST_PREDICT_PATTERN: str = os.path.join(
    PROJECT_ROOT, "predictions", "proba__raw__*.csv"
)

# Паттерн пути до сохранённого предикта меток классов на тестовом наборе
CLASSES_TEST_PREDICT_PATTERN: str = os.path.join(
    PROJECT_ROOT, "predictions", "predict__raw__*.csv"
)

# Список метрик требующих метки классов для расчета
# Используется в pred_and_metrics_compatible
CLASSES_METRIC_LIST: list[str] = ["acc"]

"""
Словари и списки признаков для управления функциями 
pipeline. Собираются в ноутбуке pipeline_config_builder.ipynb и копируются сюда.
"""
# Список признаков для загрузки из исходного датасета
PRE_FEATURES: list[str] = [
    "id",
    "rn",
    "pre_since_opened",
    "pre_since_confirmed",
    "pre_pterm",
    "pre_fterm",
    "pre_till_pclose",
    "pre_till_fclose",
    "pre_loans_credit_limit",
    "pre_loans_next_pay_summ",
    "pre_loans_outstanding",
    "pre_loans_max_overdue_sum",
    "pre_loans_credit_cost_rate",
    "pre_loans5",
    "pre_loans530",
    "is_zero_loans5",
    "is_zero_loans530",
    "pre_util",
    "pre_over2limit",
    "is_zero_over2limit",
    "enc_paym_0",
    "enc_paym_1",
    "enc_paym_2",
    "enc_paym_8",
    "enc_paym_9",
    "enc_paym_10",
    "enc_paym_24",
    "enc_loans_account_holder_type",
    "enc_loans_credit_status",
    "enc_loans_credit_type",
    "enc_loans_account_cur",
    "is_zero_loans3060",
    "is_zero_loans6090",
    "is_zero_loans90",
    "enc_paym_3",
    "enc_paym_4",
    "enc_paym_5",
    "enc_paym_6",
    "enc_paym_7",
    "enc_paym_11",
    "enc_paym_12",
    "enc_paym_13",
    "enc_paym_14",
    "enc_paym_15",
    "enc_paym_16",
    "enc_paym_17",
    "enc_paym_18",
    "enc_paym_19",
    "enc_paym_20",
    "enc_paym_21",
    "enc_paym_22",
    "enc_paym_23",
]


"""
Словарь для приведения колонок к нужным типам в препроцессинге 
(cast_columns_by_map_pipeline функция) и 
при скачивании данных в функции load_dataset
"""
CAST_TYPE_MAP: dict[str, str] = {
    "id": "int32",
    "rn": "int8",
    "pre_since_opened": "int8",
    "pre_since_confirmed": "int8",
    "pre_pterm": "int8",
    "pre_fterm": "int8",
    "pre_till_pclose": "int8",
    "pre_till_fclose": "int8",
    "pre_loans_credit_limit": "int8",
    "pre_loans_next_pay_summ": "int8",
    "pre_loans_outstanding": "int8",
    "pre_loans_max_overdue_sum": "int8",
    "pre_loans_credit_cost_rate": "int8",
    "pre_loans5": "int8",
    "pre_loans530": "int8",
    "is_zero_loans5": "int8",
    "is_zero_loans530": "int8",
    "pre_util": "int8",
    "pre_over2limit": "int8",
    "is_zero_over2limit": "int8",
    "enc_paym_0": "int8",
    "enc_paym_1": "int8",
    "enc_paym_2": "int8",
    "enc_paym_8": "int8",
    "enc_paym_9": "int8",
    "enc_paym_10": "int8",
    "enc_paym_24": "int8",
    "enc_loans_account_holder_type": "int8",
    "enc_loans_credit_status": "int8",
    "enc_loans_credit_type": "int8",
    "enc_loans_account_cur": "int8",
    "is_zero_loans3060": "int8",
    "is_zero_loans6090": "int8",
    "is_zero_loans90": "int8",
    "enc_paym_3": "int8",
    "enc_paym_4": "int8",
    "enc_paym_5": "int8",
    "enc_paym_6": "int8",
    "enc_paym_7": "int8",
    "enc_paym_11": "int8",
    "enc_paym_12": "int8",
    "enc_paym_13": "int8",
    "enc_paym_14": "int8",
    "enc_paym_15": "int8",
    "enc_paym_16": "int8",
    "enc_paym_17": "int8",
    "enc_paym_18": "int8",
    "enc_paym_19": "int8",
    "enc_paym_20": "int8",
    "enc_paym_21": "int8",
    "enc_paym_22": "int8",
    "enc_paym_23": "int8",
}


"""
Список колонок получаемых при инжениринге тип которых можно 
безопасно понизить с float64 до float32 без потери информативности
из-за округления значений.
Применяется в definite_value_proportion_features_pipeline
"""
FLOAT_DOWNCAST_COLUMNS_LIST: list[str] = [
    "pre_util_prop_3",
    "enc_loans_credit_type_prop_0",
    "pre_till_pclose_prop_10",
    "pre_util_prop_6",
    "pre_loans_outstanding_prop_1",
    "pre_loans_credit_limit_prop_2",
    "pre_loans_credit_cost_rate_prop_6",
    "pre_loans_outstanding_prop_5",
    "pre_loans_credit_cost_rate_prop_11",
    "pre_loans_credit_cost_rate_prop_4",
    "pre_loans_next_pay_summ_prop_5",
    "pre_since_opened_prop_12",
    "pre_loans_credit_limit_prop_15",
    "enc_loans_credit_type_prop_2",
    "pre_since_opened_repeated_prop",
    "pre_fterm_prop_7",
    "enc_paym_0_prop_1",
    "is_zero_over2limit_prop_1",
    "pre_since_opened_prop_8",
    "pre_loans_max_overdue_sum_prop_1",
    "pre_loans_next_pay_summ_prop_0",
    "pre_pterm_prop_6",
    "pre_since_opened_prop_19",
    "is_zero_loans5_prop_1",
    "enc_loans_account_holder_type_prop_4",
    "pre_loans_credit_limit_prop_18",
    "pre_till_fclose_prop_4",
    "pre_pterm_prop_3",
    "is_zero_loans530_prop_1",
    "enc_loans_credit_status_prop_5",
    "pre_since_confirmed_prop_4",
    "pre_fterm_prop_3",
    "pre_till_fclose_prop_3",
    "pre_till_fclose_prop_1",
    "pre_till_pclose_prop_7",
    "pre_since_confirmed_prop_7",
    "enc_paym_24_prop_1",
    "pre_over2limit_prop_17",
]

"""
Список колонок на удаление в функции
enc_paym_norm_group_sum_diff_pipeline
"""
DROP_LIST_ENC_PAYM_NORM_GROUP_SUMM_DIFF: list[str] = [
    "enc_paym_1",
    "enc_paym_2",
    "enc_paym_3",
    "enc_paym_4",
    "enc_paym_5",
    "enc_paym_6",
    "enc_paym_7",
    "enc_paym_11",
    "enc_paym_12",
    "enc_paym_13",
    "enc_paym_14",
    "enc_paym_15",
    "enc_paym_16",
    "enc_paym_17",
    "enc_paym_18",
    "enc_paym_19",
    "enc_paym_20",
    "enc_paym_21",
    "enc_paym_22",
    "enc_paym_23",
    "enc_paym_avg_1_all",
    "enc_paym_avg_2_all",
    "enc_paym_avg_0_this_year",
    "enc_paym_avg_1_this_year",
    "enc_paym_avg_0_last_year",
]

"""
Спиcок признаков исходного датасета
для создания фичей средней частотности трансформером
MeanValueFrequencyTransformer
"""
MEAN_FREQ_SOURCE_LIST: list[str] = [
    "pre_util",
    "pre_loans_credit_limit",
    "pre_since_opened",
    "pre_loans_credit_cost_rate",
    "enc_loans_credit_type",
    "pre_loans_next_pay_summ",
    "pre_since_confirmed",
    "pre_pterm",
    "enc_paym_0",
    "enc_loans_account_holder_type",
    "pre_loans530",
    "enc_paym_8",
    "pre_loans5",
    "enc_paym_10",
    "enc_loans_account_cur",
    "enc_paym_9",
]

"""
Название столбца используемого для нормализации суммы частотностей 
в трансформере MeanValueFrequencyTransformer
"""
NORMA: str = "rn_max"

"""
Список признаков для удаления в функции 
definite_value_proportion_features_pipeline
Сформирован вручную
"""
DROP_LIST_MEAN_VALUE_FREQUENCY_FEATURE: list[str] = [
    "pre_loans530",
    "enc_paym_8",
    "pre_loans5",
    "enc_paym_10",
    "enc_loans_account_cur",
    "enc_paym_9",
]

"""
Словарь пропорциональных признаков 
для функции definite_value_proportion_features_pipeline
"""
PROP_FEATURES_DICT: dict[str, list[int]] = {
    "pre_loans_next_pay_summ": [5, 0],
    "enc_paym_0": [1],
    "pre_till_fclose": [4, 3, 1],
    "enc_loans_credit_type": [0, 2],
    "is_zero_loans5": [1],
    "pre_over2limit": [17],
    "pre_loans_credit_cost_rate": [6, 11, 4],
    "pre_loans_outstanding": [1, 5],
    "enc_loans_credit_status": [5],
    "is_zero_over2limit": [1],
    "pre_fterm": [7, 3],
    "pre_loans_credit_limit": [2, 15, 18],
    "pre_loans_max_overdue_sum": [1],
    "is_zero_loans530": [1],
    "enc_paym_24": [1],
    "pre_since_opened": [12, 8, 19],
    "pre_util": [3, 6],
    "pre_since_confirmed": [4, 7],
    "pre_pterm": [6, 3],
    "enc_loans_account_holder_type": [4],
    "pre_till_pclose": [10, 7],
    "is_zero_loans3060": [1],
    "is_zero_loans6090": [1],
    "is_zero_loans90": [1],
}


"""
Список признокав удаляемых из исходного датасета
функцией drop_columns_pipeline
"""

DROP_LIST: list[str] = [
    "pre_since_opened",
    "is_zero_loans3060_prop_1",
    "is_zero_loans6090_prop_1",
    "is_zero_loans90_prop_1",
    "pre_loans_next_pay_summ",
    "enc_paym_0",
    "pre_till_fclose",
    "enc_loans_credit_type",
    "is_zero_loans5",
    "pre_over2limit",
    "pre_loans_credit_cost_rate",
    "pre_loans_outstanding",
    "enc_loans_credit_status",
    "is_zero_over2limit",
    "pre_fterm",
    "pre_loans_credit_limit",
    "pre_loans_max_overdue_sum",
    "is_zero_loans530",
    "enc_paym_24",
    "pre_util",
    "pre_since_confirmed",
    "pre_pterm",
    "enc_loans_account_holder_type",
    "pre_till_pclose",
    "is_zero_loans3060",
    "is_zero_loans6090",
    "is_zero_loans90",
]


# Константы классификатора

# Random seed для воспроизводимости
SEED: int = 0
# Количество фолдов в CatBoostEnsembleClassifier
N_SPLITS: int = 5
# Стратификация разделения фолдов в CatBoostEnsembleClassifier
SHUFFLE: bool = True
# Список категориальных фичей в CatBoostEnsembleClassifier
CAT_FEATURES: list[str] = []

"""
Список словарей гиперпараметров моделей ансамбля
подобранных Optuna
"""
PARAMS_LIST: list[dict[str, int | float | bool | str]] = [
    {
        "verbose": 0,
        "use_best_model": True,
        "random_seed": 0,
        "grow_policy": "SymmetricTree",
        "border_count": 113,
        "min_data_in_leaf": 5,
        "random_strength": 8.209932299,
        "learning_rate": 0.03625476076,
        "iterations": 3000,
        "l2_leaf_reg": 8.005778243,
        "boosting_type": "Plain",
        "od_wait": 100,
        "depth": 4,
        "subsample": 0.9882297325,
        "bagging_temperature": 0.09710127579,
        "rsm": 0.7343256008,
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "auto_class_weights": "Balanced",
    },
    {
        "verbose": 0,
        "use_best_model": True,
        "random_seed": 0,
        "grow_policy": "SymmetricTree",
        "border_count": 113,
        "min_data_in_leaf": 5,
        "random_strength": 8.209932299,
        "learning_rate": 0.03625476076,
        "iterations": 3000,
        "l2_leaf_reg": 8.005778243,
        "boosting_type": "Plain",
        "od_wait": 100,
        "depth": 4,
        "subsample": 0.9882297325,
        "bagging_temperature": 0.09710127579,
        "rsm": 0.7343256008,
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "auto_class_weights": "Balanced",
    },
    {
        "verbose": 0,
        "use_best_model": True,
        "random_seed": 0,
        "grow_policy": "SymmetricTree",
        "border_count": 113,
        "min_data_in_leaf": 5,
        "random_strength": 8.209932299,
        "learning_rate": 0.03625476076,
        "iterations": 3000,
        "l2_leaf_reg": 8.005778243,
        "boosting_type": "Plain",
        "od_wait": 100,
        "depth": 4,
        "subsample": 0.9882297325,
        "bagging_temperature": 0.09710127579,
        "rsm": 0.7343256008,
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "auto_class_weights": "Balanced",
    },
    {
        "verbose": 0,
        "use_best_model": True,
        "random_seed": 0,
        "grow_policy": "SymmetricTree",
        "border_count": 113,
        "min_data_in_leaf": 5,
        "random_strength": 8.209932299,
        "learning_rate": 0.03625476076,
        "iterations": 3000,
        "l2_leaf_reg": 8.005778243,
        "boosting_type": "Plain",
        "od_wait": 100,
        "depth": 4,
        "subsample": 0.9882297325,
        "bagging_temperature": 0.09710127579,
        "rsm": 0.7343256008,
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "auto_class_weights": "Balanced",
    },
    {
        "verbose": 0,
        "use_best_model": True,
        "random_seed": 0,
        "grow_policy": "SymmetricTree",
        "border_count": 113,
        "min_data_in_leaf": 5,
        "random_strength": 8.209932299,
        "learning_rate": 0.03625476076,
        "iterations": 3000,
        "l2_leaf_reg": 8.005778243,
        "boosting_type": "Plain",
        "od_wait": 100,
        "depth": 4,
        "subsample": 0.9882297325,
        "bagging_temperature": 0.09710127579,
        "rsm": 0.7343256008,
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "auto_class_weights": "Balanced",
    },
    {
        "iterations": 2996,
        "learning_rate": 0.036254760756236626,
        "depth": 4,
        "l2_leaf_reg": 8.005778242558318,
        "rsm": 0.7343256008238508,
        "border_count": 113,
        "random_seed": 0,
        "verbose": False,
        "auto_class_weights": "Balanced",
        "random_strength": 8.209932298658357,
        "eval_metric": "AUC",
        "bagging_temperature": 0.09710127579306127,
        "boosting_type": "Plain",
        "subsample": 0.9882297325066979,
        "early_stopping_rounds": 100,
        "grow_policy": "SymmetricTree",
        "min_data_in_leaf": 5,
    },
]

"""
Веса моделей для взвешивания финального предсказания ансамбля.
для моделей фолдов это их AUC на валидационных наборах,
для финальной модели обученной на всех данных это средний 
AUC моделей фолдов.
"""
WEIGHTS_LIST: list[float] = [
    0.7558470790046106,
    0.7541369034301536,
    0.7517121820944949,
    0.7529815101963229,
    0.7508019090111355,
    0.7530959167473436,
]

"""
Порог предсказания класса 1.
Вычисляется в roc_curve_and_treshholds_selections.ipynb 
в главе Optimal Thresholds selection.
"""
THRESHOLD: float = 0.48276
