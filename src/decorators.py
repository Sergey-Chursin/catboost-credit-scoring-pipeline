import functools
import logging
from typing import Callable
import pandas as pd

from memory_utils import (
    rss_process_statistic,
    cgroup_memory_statistic
)


def memory_monitor_function(func: Callable) -> Callable:
    """
    Декоратор для автоматического мониторинга памяти в функциях
    preprocessing и feature engineering.

    Автоматически добавляет диагностику памяти до и после выполнения функции.
    Работает только при уровне логирования DEBUG.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Получаем логгер из модуля,
        # где определена функция, передаваемая в декоратор
        logger = logging.getLogger(func.__module__)

        # Логируем название функции
        logger.info(f'FUNCTION {func.__name__}')

        # Находим DataFrame среди аргументов
        df = None
        if args and isinstance(args[0], pd.DataFrame):
            df = args[0]

        # Выводим логи диагностики RAM ДО выполнения логики функции
        if logger.isEnabledFor(logging.DEBUG) and df is not None:
            logger.debug('INCOMING statistics')
            # Логируем RSS процесса и объекты в RAM
            rss_process_statistic(df)
            # Логируем потребление памяти по cgroup
            cgroup_memory_statistic()

        # Выполняем основную функцию
        result = func(*args, **kwargs)

        # Диагностика ПОСЛЕ выполнения логики функции
        if logger.isEnabledFor(logging.DEBUG) and isinstance(result, pd.DataFrame):
            logger.debug('OUTPUT statistics')
            rss_process_statistic(result)
            cgroup_memory_statistic()

        return result

    return wrapper


def memory_monitor_transformer(cls):
    """
    Декоратор для автоматического мониторинга памяти в
    трансформерах  feature engineering.
    Оборачивает все публичные методы трансформера
    в функцию-обёртку monitor с диагностикой памяти до и после их выполнения.
    Работает только при уровне логирования DEBUG.
    cls — это декорируемый класс-трансформер sklearn-compatible.
    Возвращает модифицированный класс, где указанные методы автоматически
    дополнены логированием RAM-метрик.
    """
    monitored_methods = [
        "fit",
        "transform",
        "fit_transform",
        "predict",
        "predict_proba"
    ]
    def monitor(method_name, method):
        """
        Функция-обёртка (декоратор), которая добавляет мониторинг
        памяти вокруг исходного метода.
        """
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            # Получаем логгер из модуля,
            # где определен класс, передаваемый в декоратор.
            logger = logging.getLogger(self.__class__.__module__)
            # Логируем название класса и метода
            logger.info(f"TRANSFORMER {self.__class__.__name__}.{method_name}")

            # Находим DataFrame среди аргументов
            df = None
            if args and isinstance(args[0], pd.DataFrame):
                df = args[0]

            # Выводим логи диагностики RAM ДО выполнения логики метода
            if logger.isEnabledFor(logging.DEBUG) and df is not None:
                logger.debug('INCOMING statistics')
                # Логируем RSS процесса и объекты в RAM
                rss_process_statistic(df)
                # Логируем потребление памяти по cgroup
                cgroup_memory_statistic()

            # Выполняем логику метода
            result = method(self, *args, **kwargs)

            # Диагностика RAM ПОСЛЕ выполнения логики метода
            if logger.isEnabledFor(logging.DEBUG) and isinstance(result, pd.DataFrame):
                logger.debug('OUTPUT statistics')
                rss_process_statistic(result)
                cgroup_memory_statistic()

            return result

        return wrapper

    # Формируем словарь: ключ — имя метода, значение — ссылка на сам метод (или None, если не найден).
    # getattr(cls, name, None) - принимает имя объекта(наш класс) и название его метода в формате "str",
    # и возвращает cls.method, если такого метода нет, то возвращает третий аргумент(у нас None).
    # hasattr(cls, name) возвращает True, если у объекта действительно есть атрибут с этим именем.
    dispatcher = {name: getattr(cls, name, None) for name in monitored_methods if hasattr(cls, name)}

    # Проходим циклом по словарю
    for method_name, method in dispatcher.items():
        # Заменяем оригинальный метод (например, cls.fit) на обёрнутый (monitor('fit', cls.fit))
        # setattr —  позволяет динамически назначить новый атрибут (или заменить существующий) для объекта.
        # cls — это класс трансформера
        # method_name - имя перехватываемого метода
        # monitor(...) — функция-обёртка (декоратор)
        setattr(cls, method_name, monitor(method_name, method))

    # Возвращаем модифицированный класс
    return cls