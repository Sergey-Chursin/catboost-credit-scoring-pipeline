import gc
import logging
from typing import Any

import numpy as np
import pandas as pd
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

    Attributes:
        norma (str): Название колонки для нормализации агрегации.
        col_suffix (str): Суффикс для названий новых признаков.
        columns (list[str]): Список имен признаков для агрегации.
            Если в аргумент передан None, то атрибут становится пустым списком.
        drop_list (list[str] | None): Список признаков, которые будут удалены после обработки.
        logger (logging.Logger | None): Логгер для отладки и логирования.
        freq_maps_(dict[str, dict[int | str, float]]): Словарь, где ключи - названия колонок,
            значения - частотные словари - словари, где ключи это уникальные значения колонки,
            а значения это их частота в обучающем датасете.
        mean_freqs_(dict[str, float]): Словарь, где ключи - названия колонок, значения - средняя частота
            уникальных значений в обучающем датасете.
    """

    def __init__(
        self,
        norma: str,
        col_suffix: str = "_mean_freq",
        columns: list[str] | None = None,
        drop_list: list[str] | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Инициализирует трансформер с заданными параметрами.
        Args:
            norma (str): Название столбца с количеством записей в группе id.
                Признак отражает сколько записей есть для одного клиента.
            col_suffix (str): Суффикс для названий новых признаков.
                По умолчанию "_mean_freq".
            columns (list[str] | None): Список имен признаков для обработки.
                Если None, трансформер будет "прозрачным". По умолчанию None.
                Это позволяет "выключать" трансформер не удаляя его из пайплайна.
            drop_list (list[str] | None): Список признаков, которые будут удалены после обработки.
                По умолчанию None.
            logger (logging.Logger | None): Логгер для отладки и логирования.
                По умолчанию None.
        """
        self.norma: str = norma
        self.col_suffix: str = col_suffix
        self.columns: list[str] = columns if columns is not None else []
        self.drop_list: list[str] | None = drop_list
        self.logger: logging.Logger | None = logger
        self.freq_maps_: dict[str, dict[int | str, float]] = {}
        self.mean_freqs_: dict[str, float] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series | Any = None,
    ) -> "MeanValueFrequencyTransformer":
        """
        Обучает трансформер, вычисляя частоты значений в тренировочном датасете.

        На основе входных данных 'X' вычисляет и сохраняет в атрибуты 'freq_maps_'
        и 'mean_freqs_' необходимые для трансформации статистики.

        Args:
            X (pd.DataFrame): Входной DataFrame для обучения.
            y (np.ndarray | pd.Series | Any): Не используется, оставлен для совместимости
                с 'scikit-learn Pipeline". По умолчанию None.

        Returns:
            MeanValueFrequencyTransformer: Обученный экземпляр трансформера (self).
        """
        for col in self.columns:
            # Вычисляем относительную частоту каждого уникального значения в столбце
            self.freq_maps_[col] = X[col].value_counts(normalize=True).to_dict()
            # Вычисляем среднеарифметическое значение частотности, для заполнения им
            # новых значений, которых не было в обучающем датасете и которые могут
            # появится на новых данных.
            # Для каждой колонки берём её частотный словарь и получаем список его значений.
            # Преобразуем во float для согласования типов с анотацией.
            self.mean_freqs_[col] = float(np.mean(list(self.freq_maps_[col].values())))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет вычисленные частоты к данным.
        Создает новые признаки на основе статистик, полученных в методе '.fit()'.

        Args:
            X (pd.DataFrame): DataFrame для трансформации.

        Returns:
            pd.DataFrame: Трансформированный DataFrame с новыми признаками.
        """
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
            freq_series = X[col].map(self.freq_maps_[col]).fillna(self.mean_freqs_[col])
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

    def fit_transform(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | Any | None = None,
        **fit_params: Any,
    ) -> Any:
        """
        Объединяет обучение и трансформацию данных.

        Args:
            X (np.ndarray | pd.DataFrame): Входной DataFrame.
            y (np.ndarray | pd.Series | Any | None): Не используется. По умолчанию None.
            **fit_params (Any): Дополнительные параметры для совместимости с Pipeline.

        Returns:
            Any: Трансформированный DataFrame. По факту возвращаемый тип pd.DataFrame, но он не совместим
                с возвращаемым типом родительского класса из sklearn, поэтому выбран тип Any.
        """
        # Приводим входящий датасет к типу pd.DataFrame
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.fit(X_df, y)
        return self.transform(X_df)
