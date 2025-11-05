import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold


class CatBoostEnsembleClassifier(
    BaseEstimator,
    ClassifierMixin,
):
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
        params_list (list[dict]): Список словарей с гиперпараметрами для каждой модели ансамбля
                (N фолдов + 1 финальная модель).
        weights_list (list[float]): Список весов для взвешивания предиктов
            (один на каждую модель в ансамбле).
        threshold (float): Порог отсечения для жёсткой классификации (predict).
        cat_features (list[str]): Список названий категориальных фичей.
            Если не передан, используется пустой список.
        n_splits (int): Количество фолдов для разбиения данных (StratifiedKFold).
        seed (int): Seed для воспроизводимости разбиения и моделей.
        shuffle (bool): Флаг перемешивания данных при разбиении на фолды.
        logger (logging.Logger | None): Объект логгера для сообщений внутренней работы классификатора.
        models_ (list[CatBoostClassifier]): Список обученных моделей CatBoostClassifier
            после вызова fit.

    """

    def __init__(
        self,
        params_list: list[dict],
        weights_list: list[float],
        threshold: float = 0.5,
        cat_features: list[str] | None = None,
        n_splits: int = 5,
        seed: int = 0,
        shuffle: bool = True,
        logger: logging.Logger | None = None,
    ):
        """
        Args:
            params_list (list[dict]): Список словарей с гиперпараметрами для каждой модели ансамбля
                (N фолдов + 1 финальная модель).
            weights_list (list[float]): Список весов для взвешивания предиктов
                (один на каждую модель в ансамбле).
            threshold (float): Порог отсечения для жёсткой классификации (predict).
                По умолчанию 0.5.
            cat_features (list[str] | None): Список названий категориальных фичей.
                По умолчанию None.
            n_splits (int): Количество фолдов для разбиения данных (StratifiedKFold).
                По умолчанию 5.
            seed (int): Seed для воспроизводимости разбиения и моделей.
                 По умолчанию 0.
            shuffle (bool): Флаг перемешивания данных при разбиении на фолды.
                По умолчанию True.
            logger (logging.Logger | None): Объект логгера для сообщений внутренней работы классификатора.
                По умолчанию None.
        """
        self.params_list: list[dict] = params_list
        self.weights_list: list[float] = weights_list
        self.threshold: float = threshold
        self.cat_features: list[str] = cat_features if cat_features is not None else []
        self.n_splits: int = n_splits
        self.seed: int = seed
        self.shuffle: bool = shuffle
        self.logger: logging.Logger | None = logger
        self.models_: list[CatBoostClassifier] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "CatBoostEnsembleClassifier":
        """
        Обучает ансамбль моделей на разных фолдах и финальную модель
        на полном наборе данных.
        Args:
            X (pd.DataFrame): Тренировочный датафрейм.
            y (pd.Series): Целевой признак.
        Returns:
            CatBoostEnsembleClassifier : Обученный объект классификатора с атрибутом models_,
              содержащим список обученных моделей.
        """
        if self.logger is not None:
            self.logger.info("CLASSIFIER fit")

        self.models_ = []
        kf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.seed,
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
                data=X_train,
                label=y_train,
                cat_features=self.cat_features,
            )
            val_pool = Pool(
                data=X_val,
                label=y_val,
                cat_features=self.cat_features,
            )
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
            model.fit(
                train_pool,
                eval_set=val_pool,
            )
            self.models_.append(model)

        # Обучаем финальную модель на полном наборе данных
        if self.logger is not None:
            self.logger.info("Fit final model")
        train_pool = Pool(
            data=X,
            label=y,
            cat_features=self.cat_features,
        )
        params = self.params_list[self.n_splits]
        # Выводим гиперпараметры в лог
        if self.logger is not None:
            self.logger.info("Params for final model  %s", params)

        model = CatBoostClassifier(**params)
        model.fit(train_pool)
        self.models_.append(model)
        return self

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Возвращает датафрейм без изменений.
        Метод добавлен для совместимости с Pipeline.
        Args:
            X (pd.DataFrame): Пандас датафрейм.
        Returns:
            pd.DataFrame: Исходный датафрейм без изменений.
        """
        if self.logger is not None:
            self.logger.info("CLASSIFIER fit_transform")
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Возвращает датафрейм без изменений.
        Метод добавлен для совместимости с Pipeline.
        Args:
            X (pd.DataFrame): Пандас датафрейм.
        Returns:
            pd.DataFrame: Исходный датафрейм без изменений.
        """
        if self.logger is not None:
            self.logger.info("CLASSIFIER transform")
        return X

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказывает вероятности классов,
        усреднённые по всем моделям с учётом весов.
        Args:
            X (pd.DataFrame): Пандас датафрейм.
        Returns:
            np.ndarray: Массив предсказанных вероятностей для классов 0 и 1,
            размерностью (n_samples, 2).
        """
        if self.logger is not None:
            self.logger.info("CLASSIFIER predict_proba")

        preds = []
        for model, weight in zip(
            self.models_,
            self.weights_list,
            strict=True,
        ):
            pred = model.predict_proba(X)[:, 1]
            preds.append(pred * weight)
        mean_pred = np.sum(preds, axis=0) / np.sum(self.weights_list)
        return np.vstack([1 - mean_pred, mean_pred]).T

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказывает классы на основе вероятностей с порогом 0.5
        по умолчанию либо с переданным, например
        при котором разница между TPR и FPR максимальна,
        либо подобранным с учётом бизнес логики.
        Args:
            X (pd.DataFrame): Пандас датафрейм.
        Returns:
            np.ndarray: Массив предсказанных классов (0 или 1).
        """
        if self.logger is not None:
            self.logger.info("CLASSIFIER predict")
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
