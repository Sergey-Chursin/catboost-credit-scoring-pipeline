import logging
from typing import Dict, List, Optional

import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold


class CatBoostEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Бинарный ансамблевый классификатор на основе CatBoost, обучающий N моделей
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
        params_list: List[Dict],
        weights_list: List[float],
        threshold: float = 0.5,
        cat_features: Optional[List[str]] = None,
        n_splits: int = 5,
        seed: int = 0,
        shuffle: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            params_list (List[Dict]): Список словарей с гиперпараметрами для каждой модели ансамбля
                (N фолдов + 1 финальная модель). REQUIRED.
            weights_list (List[float]): Список весов для взвешивания предиктов
                (один на каждую модель в ансамбле). REQUIRED.
            threshold (float, optional): Порог отсечения для жёсткой классификации (predict).
                По умолчанию 0.5.
            cat_features (List[str], optional): Список названий категориальных фичей.
                Если не передан, используется пустой список.
            n_splits (int, optional): Количество фолдов для разбиения данных (StratifiedKFold).
                По умолчанию 5.
            seed (int, optional): Seed для воспроизводимости разбиения и моделей.
                 По умолчанию 0.
            shuffle (bool, optional): Флаг перемешивания данных при разбиении на фолды.
                По умолчанию True.
            logger (logging.Logger, optional): Объект логгера для сообщений внутренней работы классификатора.
                По умолчанию None (без логирования).
        """
        self.params_list = params_list
        self.weights_list = weights_list
        self.threshold = threshold
        self.cat_features = cat_features if cat_features is not None else []
        self.n_splits = n_splits
        self.seed = seed
        self.shuffle = shuffle
        self.logger = logger

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
            self.logger.info("CLASSIFIER fit")

        self.models_ = []
        kf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed
        )

        # Обучаем N моделей на разных фолдах
        for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            # Выводим номер фолда в лог
            if self.logger is not None:
                self.logger.info("Fit fold %s", i)

            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            train_pool = Pool(
                data=X_train, label=y_train, cat_features=self.cat_features
            )
            val_pool = Pool(data=X_val, label=y_val, cat_features=self.cat_features)
            """
            Для обучения используются валидационные подвыборки, 
            для остановки обучения вместо ГП early_stopping_rounds
            в params передаётся od_wait.
            """
            params = self.params_list[i]

            # Выводим гиперпараметры в лог
            if self.logger is not None:
                self.logger.info("Params for fold %s", params)

            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool)
            self.models_.append(model)

        # Обучаем финальную модель на полном наборе данных
        if self.logger is not None:
            self.logger.info("Fit final model")
        train_pool = Pool(data=X, label=y, cat_features=self.cat_features)
        params = self.params_list[self.n_splits]
        # Выводим гиперпараметры в лог
        if self.logger is not None:
            self.logger.info("Params for final model  %s", params)

        model = CatBoostClassifier(**params)
        model.fit(train_pool)
        self.models_.append(model)
        return self

    def fit_transform(self, X, y=None):
        if self.logger is not None:
            self.logger.info("CLASSIFIER fit_transform")
        # Обучаем классификатор
        self.fit(X, y)
        # Возвращаем X без изменений
        return X

    def transform(self, X):
        if self.logger is not None:
            self.logger.info("CLASSIFIER transform")
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
            self.logger.info("CLASSIFIER predict_proba")

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
            self.logger.info("CLASSIFIER predict")
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
