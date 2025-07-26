from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold


from src.config import (
    CAT_FEATURES,
    N_SPLIT,
    SEED,
    SHUFFLE,
    THRESHOLD,
    PARAMS_LIST,
    WEIGHTS_LIST
)

class CatBoostEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ансамблевый классификатор на основе CatBoost, обучающий N моделей
    на разных разбиениях данных и финальную модель на полном наборе данных.
    Чтобы повысить устойчивость и качество предсказаний
    за счёт усреднения результатов.

    Особенности:
        - Используется StratifiedKFold для разбиения данных на N фолдов.
        - Для каждого фолда обучается отдельная модель на тренировочной части.
        - Все N моделей сохраняются для последующего усреднения предсказаний.
        - Дополнительно обучается финальная модель на полном наборе данных.
        - Финальные предсказания вероятности классов это средневзвешенные
          предсказания моделей ансамбля через их веса (AUC или другая метрика)

    Args:
       params_list (list of dict, optional): Список параметров для каждой модели
            (N фолдов + 1 финальная модель).
            Приоритет: аргумент > globals() (PARAMS_LIST) > error (обязательный).
        weights_list (list of float, optional): Веса для усреднения предсказаний.
            (N фолдов + 1 финальная модель).
            Приоритет: аргумент > globals() (WEIGHTS_LIST) > error (обязательный).
        threshold (float, optional): Порог для классификации.
            Приоритет: аргумент > globals() (THRESHOLD) > 0.5.
        cat_features (list, optional): Список категориальных фичей.
            Приоритет: аргумент > globals() (CAT_FEATURES) > [].
        n_splits (int, optional): Количество фолдов.
            Приоритет: аргумент > globals() (N_SPLIT) > 5.
        seed (int, optional): Random seed.
            Приоритет: аргумент > globals() (SEED) > 0.
        shuffle (bool, optional): Перемешивание в KFolds.
            Приоритет: аргумент > globals() (SHUFFLE) > True.
        logger (logging.Logger, optional): Объект логгера.
            Приоритет: аргумент  > None.

    Attributes:
        models_ (list): Список обученных моделей CatBoostClassifier
            после вызова fit.

    Methods:
        fit(X, y): Обучает ансамбль моделей и сохраняет их.
        fit_transform(X, y): Обучает модели и возвращает X без изменений
            (для совместимости с пайплайнами).
        transform(X): Возвращает X без изменений
            (для совместимости с пайплайнами).
        predict_proba(X): Возвращает взвешенное усреднённое предсказание
            вероятностей положительного класса.
        predict(X): Возвращает бинарные предсказания
            с порогом 0.5 по умолчанию или переданному.
    """
    def __init__(
            self,
            params_list=None,
            weights_list=None,
            threshold=None,
            cat_features=None,
            n_splits=None,
            seed=None,
            shuffle=None,
            logger=None
    ):
        if params_list is not None:
            self.params_list = params_list
        elif 'PARAMS_LIST' in globals():
            self.params_list = PARAMS_LIST
        else:
            raise ValueError('params_list must be provided or defined in config (PARAMS_LIST)')

        if weights_list is not None:
            self.weights_list = weights_list
        elif 'WEIGHTS_LIST' in globals():
            self.weights_list = WEIGHTS_LIST
        else:
            raise ValueError('weights_list must be provided or defined in config (WEIGHTS_LIST)')

        if threshold is not None:
            self.threshold = threshold
        elif 'THRESHOLD' in globals():
            self.threshold = THRESHOLD
        else:
            self.threshold = 0.5

        if cat_features is not None:
            self.cat_features = cat_features
        elif 'CAT_FEATURES' in globals():
            self.cat_features = CAT_FEATURES
        else:
            self.cat_features = []

        if n_splits is not None:
            self.n_splits = n_splits
        elif 'N_SPLIT' in globals():
            self.n_splits = N_SPLIT
        else:
            self.n_splits = 5

        if seed is not None:
            self.seed = seed
        elif 'SEED' in globals():
            self.seed = SEED
        else:
            self.seed = 0

        if shuffle is not None:
            self.shuffle = shuffle
        elif 'SHUFFLE' in globals():
            self.shuffle = SHUFFLE
        else:
            self.shuffle = True

        if logger is not None:
            self.logger = logger
        else:
            self.logger = None

    def fit(self, X, y):
        """
        Обучает ансамбль моделей на разных фолдах и финальную модель
        на полном наборе данных.

        Args:
            X (pd.DataFrame): Признаки.
            y (pd.Series или np.array): Целевой признак.

        Returns:
            self : Обученный объект классификатора с атрибутом models_,
              содержащим список обученных моделей.
        """
        if self.logger is not None:
            self.logger.info('CLASSIFIER fit')


        self.models_ = []
        kf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.seed
        )

        # Обучаем N моделей на разных фолдах
        for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            # Выводим номер фолда в лог
            if self.logger is not None:
                self.logger.info('Fit fold %s', i)

            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            train_pool = Pool(
                data=X_train,
                label=y_train,
                cat_features=self.cat_features
            )
            val_pool = Pool(
                data=X_val,
                label=y_val,
                cat_features=self.cat_features
            )
            """
            Для обучения используются валидационные подвыборки, 
            для остановки обучения вместо ГП early_stopping_rounds
            в params передаётся od_wait.
            """
            params = self.params_list[i]

            # Выводим гиперпараметры в лог
            if self.logger is not None:
                self.logger.info('Params for fold %s', params)

            model = CatBoostClassifier(
                **params
            )
            model.fit(
                train_pool,
                eval_set=val_pool
            )
            self.models_.append(model)

        # Обучаем финальную модель на полном наборе данных
        if self.logger is not None:
            self.logger.info('Fit final model')
        train_pool = Pool(
            data=X,
            label=y,
            cat_features=self.cat_features
        )
        params = self.params_list[self.n_splits]
        # Выводим гиперпараметры в лог
        if self.logger is not None:
            self.logger.info('Params for final model  %s', params)

        model = CatBoostClassifier(
            **params
        )
        model.fit(
            train_pool
        )
        self.models_.append(model)
        return self

    def fit_transform(self, X, y=None):
        if self.logger is not None:
            self.logger.info('CLASSIFIER fit_transform')
        # Обучаем классификатор
        self.fit(X, y)
        # Возвращаем X без изменений
        return X

    def transform(self, X):
        if self.logger is not None:
            self.logger.info('CLASSIFIER transform')
        # Возвращаем X без изменений
        return X

    def predict_proba(self, X):
        """
        Предсказывает вероятности классов,
        усреднённые по всем моделям с учётом весов.

        Args:
            X (pd.DataFrame): Матрица признаков.

        Returns:
            np.ndarray: Массив вероятностей для классов 0 и 1,
            размерностью (n_samples, 2).
        """
        if self.logger is not None:
            self.logger.info('CLASSIFIER predict_proba')

        preds = []
        for model, weight in zip(self.models_, self.weights_list):
            pred = model.predict_proba(X)[:, 1]
            preds.append(pred * weight)
        mean_pred = np.sum(preds, axis=0) / np.sum(self.weights_list)
        return np.vstack([1 - mean_pred, mean_pred]).T

    def predict(self, X):
        """
        Предсказывает классы на основе вероятностей с порогом 0.5
        по умолчанию либо с переданным, например
        при котором разница между TPR и FPR максимальна,
        либо подобранным с учётом бизнес логики.

        Args:
            X (pd.DataFrame): Матрица признаков.

        Returns:
            np.ndarray: Массив предсказанных классов (0 или 1).
        """
        if self.logger is not None:
            self.logger.info('CLASSIFIER predict')
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)