import functools
import logging
from typing import Callable
import pandas as pd

from memory_utils import (
    rss_process_statistic,
    cgroup_memory_statistic
)


def memory_monitor(func: Callable) -> Callable:
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