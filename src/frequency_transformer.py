import gc
import logging
from typing import Any, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.decorators import memory_monitor_transformer


@memory_monitor_transformer
class MeanValueFrequencyTransformer(BaseEstimator, TransformerMixin):
    """
    Трансформер для генерации новых признаков, отражающих среднюю частоту (относительную встречаемость)
    значений указанных столбцов по каждой группе id.

    Для каждого признака из списка columns создаёт новый столбец вида {column}_mean_freq,
    содержащий отношение суммы частот значений по группе id к количеству записей в этой группе (norma).
    Новые значения, не встречавшиеся в обучающем датасете, получают среднюю частоту по обучающей выборке.

    Args:
        norma (str): Название столбца с количеством записей в группе id.
        col_suffix (str): Постфикс для новых признаков.
        columns (Optional[List[str]], optional): Список имен признаков для агрегации.
        drop_list (List[str]): Список признаков, которые будут удалены после обработки.
        logger (Optional[logging.Logger], optional): Логгер для отладки и логирования.

    Attributes:
        freq_maps(dict): Словарь где ключи - названия колонок, значения - частотные словари.
        mean_freqs(dict): Словарь где ключи - названия колонок, значения - средняя частота.
    """

    def __init__(
        self,
        norma: str,
        col_suffix: str = "_mean_freq",
        columns: Optional[List[str]] = None,
        drop_list: List[str] | None = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.norma = norma
        self.col_suffix = col_suffix
        self.columns = columns
        self.drop_list = drop_list
        self.logger = logger
        self.freq_maps: dict[str, Any] = {}
        self.mean_freqs: dict[str, Any] = {}

    def fit(self, X, y=None):
        # if self.logger is not None:
        #     self.logger.info('FREQUENCY TRANSFORMER fit')
        for col in self.columns:
            # Вычисляем относительную частоту каждого уникального значения в столбце
            self.freq_maps[col] = X[col].value_counts(normalize=True).to_dict()
            # Вычисляем среднеарифметическое значение частотности, для заполнения им
            # новых значений, которых не было в обучающем датасете и которые могут
            # появится на новых данных.
            self.mean_freqs[col] = np.mean(list(self.freq_maps[col].values()))
        return self

    def transform(self, X):
        # if self.logger is not None:
        #     self.logger.info('FREQUENCY TRANSFORMER transform')
        if self.logger is not None:
            self.logger.info("NEW features")
        # проходим циклом по колонкам из списка
        for col in self.columns:
            new_col = f"{col}{self.col_suffix}"
            if self.logger is not None:
                self.logger.info(new_col)
            # Создаём Series с частотами значений для каждой строки
            # Новые значения, не входившие в тренировочный датасет,
            # заполняем средней частотой
            freq_series = X[col].map(self.freq_maps[col]).fillna(self.mean_freqs[col])
            # Делаем группировку столбца по id и считаем сумму частот в группе,
            # делим сумму на количество записей для этого id.
            # Результат сохраняем в новый столбец new_col.
            X[new_col] = freq_series.groupby(X["id"]).transform("sum") / X[self.norma]

            # Удаляем временную переменную для экономии памяти
            del freq_series
            gc.collect()

        # Если передан список колонок на удаление
        # то даляем уже не нужные колонки.
        if self.drop_list is not None:
            X = X.drop(self.drop_list, axis=1)
            if self.logger is not None:
                self.logger.info(f"DataFrame shape after drop(): {X.shape}")

        return X

    def fit_transform(self, X, y=None):
        # if self.logger is not None:
        #     self.logger.info('FREQUENCY TRANSFORMER fit_transform')
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        # if self.logger is not None:
        #     self.logger.info('FREQUENCY TRANSFORMER predict')
        return self.transform(X)

    def predict_proba(self, X):
        # if self.logger is not None:
        #     self.logger.info('FREQUENCY TRANSFORMER predict_proba')
        return self.transform(X)
