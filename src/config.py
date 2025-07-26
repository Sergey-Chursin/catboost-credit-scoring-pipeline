import os

# Модуль с константами, параметрами, путями, списками и словарями для пайплайна.


# Random seed для воспроизводимости
SEED = 0
# Количество фолдов в CatBoostEnsembleClassifier
N_SPLIT = 5
# Стратификация разделения фолдов в CatBoostEnsembleClassifier
SHUFFLE = True
# Список категориалбных фичей в CatBoostEnsembleClassifier
CAT_FEATURES = []

# Пути к директориям (относительные от корня проекта)
MODELS_PATH = '../models/'  # Путь к сохранённым моделям (.bin, .pkl)
PROCESSED_DATA_PATH = '../processed_data/'  # Путь к подготовленным данным (X_train.csv и т.д.)
PREDICTIONS_PATH = '../predictions/'  # Путь к предсказаниям (test_predict.pkl)

"""
Словари и списки признаков для управления функциями 
pipeline. Собираются в pipeline_config_builder.ipynb 
и копируются сюда.
"""

# Список признаков для загрузки из исходного датасета
PRE_FEATURES = [
    'id',
    'rn',
    'pre_since_opened',
    'pre_since_confirmed',
    'pre_pterm',
    'pre_fterm',
    'pre_till_pclose',
    'pre_till_fclose',
    'pre_loans_credit_limit',
    'pre_loans_next_pay_summ',
    'pre_loans_outstanding',
    'pre_loans_max_overdue_sum',
    'pre_loans_credit_cost_rate',
    'pre_loans5',
    'pre_loans530',
    'is_zero_loans5',
    'is_zero_loans530',
    'pre_util',
    'pre_over2limit',
    'is_zero_over2limit',
    'enc_paym_0',
    'enc_paym_1',
    'enc_paym_2',
    'enc_paym_8',
    'enc_paym_9',
    'enc_paym_10',
    'enc_paym_24',
    'enc_loans_account_holder_type',
    'enc_loans_credit_status',
    'enc_loans_credit_type',
    'enc_loans_account_cur',
    'is_zero_loans3060',
    'is_zero_loans6090',
    'is_zero_loans90',
    'enc_paym_3',
    'enc_paym_4',
    'enc_paym_5',
    'enc_paym_6',
    'enc_paym_7',
    'enc_paym_11',
    'enc_paym_12',
    'enc_paym_13',
    'enc_paym_14',
    'enc_paym_15',
    'enc_paym_16',
    'enc_paym_17',
    'enc_paym_18',
    'enc_paym_19',
    'enc_paym_20',
    'enc_paym_21',
    'enc_paym_22',
    'enc_paym_23'
]

"""
Словарь пропорциональных признаков 
для функции definite_value_proportion_features_pipeline
"""
PROP_FEATURES_DICT = {
    'pre_loans_next_pay_summ': [5, 0],
    'enc_paym_0': [1],
    'pre_till_fclose': [4, 3, 1],
    'enc_loans_credit_type': [0, 2],
    'is_zero_loans5': [1],
    'pre_over2limit': [17],
    'pre_loans_credit_cost_rate': [6, 11, 4],
    'pre_loans_outstanding': [1, 5],
    'enc_loans_credit_status': [5],
    'is_zero_over2limit': [1],
    'pre_fterm': [7, 3],
    'pre_loans_credit_limit': [2, 15, 18],
    'pre_loans_max_overdue_sum': [1],
    'is_zero_loans530': [1],
    'enc_paym_24': [1],
    'pre_since_opened': [12, 8, 19],
    'pre_util': [3, 6],
    'pre_since_confirmed': [4, 7],
    'pre_pterm': [6, 3],
    'enc_loans_account_holder_type': [4],
    'pre_till_pclose': [10, 7],
    'is_zero_loans3060': [1],
    'is_zero_loans6090': [1],
    'is_zero_loans90': [1]
}


"""
Спиcок признаков исходного датасета
для создания фичей средней частотности функцией
mean_value_frequency_feature_pipeline
"""
MEAN_FREQ_SOURCE_LIST = [
    'pre_util',
    'pre_loans_credit_limit',
    'pre_since_opened',
    'pre_loans_credit_cost_rate',
    'enc_loans_credit_type',
    'pre_loans_next_pay_summ',
    'pre_since_confirmed',
    'pre_pterm',
    'enc_paym_0',
    'enc_loans_account_holder_type',
    'pre_loans530',
    'enc_paym_8',
    'pre_loans5',
    'enc_paym_10',
    'enc_loans_account_cur',
    'enc_paym_9'
]

"""
Список признокав удаляемых из исходного датасета
функцией drop_columns_drop_duplicates_pipeline
"""
DROP_LIST = [
    'id',
    'rn',
    'pre_since_opened',
    'pre_since_confirmed',
    'pre_pterm',
    'pre_fterm',
    'pre_till_pclose',
    'pre_till_fclose',
    'pre_loans_credit_limit',
    'pre_loans_next_pay_summ',
    'pre_loans_outstanding',
    'pre_loans_max_overdue_sum',
    'pre_loans_credit_cost_rate',
    'pre_loans5',
    'pre_loans530',
    'is_zero_loans5',
    'is_zero_loans530',
    'pre_util',
    'pre_over2limit',
    'is_zero_over2limit',
    'enc_paym_0',
    'enc_paym_1',
    'enc_paym_2',
    'enc_paym_8',
    'enc_paym_9',
    'enc_paym_10',
    'enc_paym_24',
    'enc_loans_account_holder_type',
    'enc_loans_credit_status',
    'enc_loans_credit_type',
    'enc_loans_account_cur',
    'is_zero_loans3060',
    'is_zero_loans6090',
    'is_zero_loans90',
    'enc_paym_3',
    'enc_paym_4',
    'enc_paym_5',
    'enc_paym_6',
    'enc_paym_7',
    'enc_paym_11',
    'enc_paym_12',
    'enc_paym_13',
    'enc_paym_14',
    'enc_paym_15',
    'enc_paym_16',
    'enc_paym_17',
    'enc_paym_18',
    'enc_paym_19',
    'enc_paym_20',
    'enc_paym_21',
    'enc_paym_22',
    'enc_paym_23',
    'enc_paym_avg_1_all',
    'enc_paym_avg_2_all',
    'enc_paym_avg_0_this_year',
    'enc_paym_avg_1_this_year',
    'enc_paym_avg_0_last_year',
    'is_zero_loans3060_prop_1',
    'is_zero_loans6090_prop_1',
    'is_zero_loans90_prop_1'
]

"""
Список словарей гиперпараметров моделей ансамбля
подобранных Optuna
"""
PARAMS_LIST = [
    {
        'verbose': 0,
        'use_best_model': True,
        'random_seed': 0,
        'grow_policy': 'SymmetricTree',
        'border_count': 113,
        'min_data_in_leaf': 5,
        'random_strength': 8.209932299,
        'learning_rate': 0.03625476076,
        'iterations': 3000,
        'l2_leaf_reg': 8.005778243,
        'boosting_type': 'Plain',
        'od_wait': 100,
        'depth': 4,
        'subsample': 0.9882297325,
        'bagging_temperature': 0.09710127579,
        'rsm': 0.7343256008,
        'eval_metric': 'AUC',
        'loss_function': 'Logloss',
        'auto_class_weights': 'Balanced'
    },
    {
        'verbose': 0,
        'use_best_model': True,
        'random_seed': 0,
        'grow_policy': 'SymmetricTree',
        'border_count': 113,
        'min_data_in_leaf': 5,
        'random_strength': 8.209932299,
        'learning_rate': 0.03625476076,
        'iterations': 3000,
        'l2_leaf_reg': 8.005778243,
        'boosting_type': 'Plain',
        'od_wait': 100,
        'depth': 4,
        'subsample': 0.9882297325,
        'bagging_temperature': 0.09710127579,
        'rsm': 0.7343256008,
        'eval_metric': 'AUC',
        'loss_function': 'Logloss',
        'auto_class_weights': 'Balanced'
    },
    {
        'verbose': 0,
        'use_best_model': True,
        'random_seed': 0,
        'grow_policy': 'SymmetricTree',
        'border_count': 113,
        'min_data_in_leaf': 5,
        'random_strength': 8.209932299,
        'learning_rate': 0.03625476076,
        'iterations': 3000,
        'l2_leaf_reg': 8.005778243,
        'boosting_type': 'Plain',
        'od_wait': 100,
        'depth': 4,
        'subsample': 0.9882297325,
        'bagging_temperature': 0.09710127579,
        'rsm': 0.7343256008,
        'eval_metric': 'AUC',
        'loss_function': 'Logloss',
        'auto_class_weights': 'Balanced'
    },
    {
        'verbose': 0,
        'use_best_model': True,
        'random_seed': 0,
        'grow_policy': 'SymmetricTree',
        'border_count': 113,
        'min_data_in_leaf': 5,
        'random_strength': 8.209932299,
        'learning_rate': 0.03625476076,
        'iterations': 3000,
        'l2_leaf_reg': 8.005778243,
        'boosting_type': 'Plain',
        'od_wait': 100,
        'depth': 4,
        'subsample': 0.9882297325,
        'bagging_temperature': 0.09710127579,
        'rsm': 0.7343256008,
        'eval_metric': 'AUC',
        'loss_function': 'Logloss',
        'auto_class_weights': 'Balanced'
    },
    {
        'verbose': 0,
        'use_best_model': True,
        'random_seed': 0,
        'grow_policy': 'SymmetricTree',
        'border_count': 113,
        'min_data_in_leaf': 5,
        'random_strength': 8.209932299,
        'learning_rate': 0.03625476076,
        'iterations': 3000,
        'l2_leaf_reg': 8.005778243,
        'boosting_type': 'Plain',
        'od_wait': 100,
        'depth': 4,
        'subsample': 0.9882297325,
        'bagging_temperature': 0.09710127579,
        'rsm': 0.7343256008,
        'eval_metric': 'AUC',
        'loss_function': 'Logloss',
        'auto_class_weights': 'Balanced'
    },
    {
        'iterations': 2996,
        'learning_rate': 0.036254760756236626,
        'depth': 4,
        'l2_leaf_reg': 8.005778242558318,
        'rsm': 0.7343256008238508,
        'border_count': 113,
        'random_seed': 0,
        'verbose': False,
        'auto_class_weights': 'Balanced',
        'random_strength': 8.209932298658357,
        'eval_metric': 'AUC',
        'bagging_temperature': 0.09710127579306127,
        'boosting_type': 'Plain',
        'subsample': 0.9882297325066979,
        'early_stopping_rounds': 100,
        'grow_policy': 'SymmetricTree',
        'min_data_in_leaf': 5
    }
]

"""
Веса моделей для взвешивания финального предсказания ансамбля.
для моделей фолдов это их AUC на валидационных наборах,
для финальной модели обученной на всех данных это средний 
AUC моделей фолдов.
"""

WEIGHTS_LIST = [
    0.7576036850511159,
    0.7554545982995526,
    0.7532810994057619,
    0.7546988571803108,
    0.7524269260453276,
    0.7546930331964138
]

"""
Порог предсказания класса 1.
Вычисляется в roc_curve_and_treshholds_selections.ipynb 
в главе Optimal Thresholds selection.
"""
THRESHOLD = 0.4964
