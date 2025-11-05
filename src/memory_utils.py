import ctypes
import gc
import logging

import pandas as pd
import psutil

"""
Модуль предоставляет инструменты мониторинга и оптимизации использования 
оперативной памяти для выполнения пайплайна в Docker контейнерах.

Содержит функции:
Мониторинга - отслеживание потребления памяти процессом и объектами Python
Оптимизации - управление памятью для предотвращения превышения лимитов контейнера
"""

"""
Создаём локальный логгер для этого модуля
Настройки (уровень логирования, формат сообщений) наследуются от root logger, 
который обычно конфигурируется в главном файле проекта (pipeline.py).
"""
logger = logging.getLogger(__name__)


def rss_process_statistic(df: pd.DataFrame) -> None:
    """
    Логирует подробную статистику по использованию памяти и фрагментации DataFrame,
    включая размеры всех DataFrame и Series, находящихся в памяти процесса.
    Выполняется только при уровне логирования DEBUG.

    Выводит в лог:
        - Текущий объём оперативной памяти (RSS), занимаемый процессом,
            измеряет в формате MiB(мебибайты) для согласованности с
            выводом утилит мониторинга RAM.
        - Степень фрагментации DataFrame: количество блоков памяти.
        - Список всех DataFrame в памяти:
            - Размер (shape)
            - Идентификатор объекта (id)
        - Список всех Series в памяти:
            - Имя Series
            - Идентификатор объекта (id)
            - Количество Series
    Args:
        df (pd.DataFrame): Анализируемый DataFrame, для которого выводится статистика.

    Returns:
        None
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    # Логируем общий RSS процесса
    logger.debug(f"RSS: {psutil.Process().memory_info().rss / 1024**2:.2f} MiB")
    # Логируем степень фрагментации DataFrame: сколько физических блоков он занимает в памяти.
    logger.debug(
        f"DataFrame fragmentation: number of memory blocks = {df._mgr.nblocks}"
    )

    # Логируем все DataFrame в RAM: размер (shape), id.
    for obj in gc.get_objects():
        if isinstance(obj, pd.DataFrame):
            logger.debug(f"DataFrame in RAM: shape = {obj.shape}, id = {id(obj)}")

    # Логируем все Series в RAM: имя, id.
    # Подсчитываем количество Series
    series_count = 0
    for obj in gc.get_objects():
        if isinstance(obj, pd.Series):
            logger.debug(f"Series in RAM: id = {id(obj)}, name = {obj.name}")
            series_count += 1
    logger.debug(f"Number of series in RAM: {series_count}")


def cgroup_memory_statistic() -> None:
    """
    Логирует метрики памяти и swap контейнера Docker через cgroup v2
    в формате MiB(мебибайты) для согласованности с
    выводом утилит мониторинга RAM.

    Читаемые метрики:
        - memory.current: текущее потребление физической памяти
        - memory.swap.current: текущее потребление swap
    Raises:
        Exception: При ошибках чтения файлов cgroup (файлы не найдены,
                  нет прав доступа, некорректный формат данных)
    Note:
        Работает только в Linux среде с поддержкой cgroup v2.
        В других ОС или при отсутствии cgroup логирует ошибку.
        Выполняется только при уровне логирования DEBUG.
    Returns:
        None
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    try:
        # cgroup в Docker это все процессы контейнера.
        # Читаем файл /proc/self/cgroup чтобы узнать путь к нашей cgroup
        # Файл содержит строку вида: "0::/docker/1a2b3c4d..." для Docker контейнера
        with open("/proc/self/cgroup", "rt") as f:
            cgroup_path = None

            # Ищем строку для cgroup v2
            for line in f:
                # Разбиваем строку по двоеточию: "0::/path" -> ["0", "", "/path"]
                parts = line.strip().split(":")
                if len(parts) == 3 and parts[1] == "":
                    # parts[0] = "0" (hierarchy ID для cgroup v2)
                    # parts[1] = "" (пустой список контроллеров для unified)
                    # parts[2] = путь к cgroup относительно /sys/fs/cgroup
                    cgroup_path = parts[2]
                    break

    except (FileNotFoundError, PermissionError):
        logger.debug(
            "cgroup v2 not detected, memory metrics unavailable\n"
            "This function is intended for Docker containers with cgroup v2"
        )
        return

    # Проверяем совместимость
    if cgroup_path is None:
        logger.debug("cgroup v2 not detected, memory metrics unavailable")
        return

    # Строим полный путь к директории нашей cgroup
    # Linux автоматически создает файлы с метриками в /sys/fs/cgroup/
    cgroup_dir = f"/sys/fs/cgroup{cgroup_path}"

    # Вспомогательная функция для безопасного чтения файлов
    def read_file(path: str) -> str | None:
        """
        Читает содержимое файла и возвращает строку или None при ошибке.
         Args:
             path (str): Путь к файлу
         Return:
             str | None: Содержимое файла в виде строки, либо None при ошибке.
        """
        try:
            with open(path, "rt") as f:
                return f.read().strip()

        except FileNotFoundError:
            logger.debug(f"File not found: {path}")
            return None

        except PermissionError:
            logger.debug(f"Permission denied: {path}")
            return None

        except Exception as e:
            logger.debug(f"Unexpected error reading {path}: {e}")
            return None

    # Читаем файлы с метриками памяти
    # Текущая память cgroup
    current = read_file(f"{cgroup_dir}/memory.current")
    # Текущий swap cgroup
    swap = read_file(f"{cgroup_dir}/memory.swap.current")

    # Конвертируем байты в MiB и выводим результат
    if current and current.isdigit():
        current_mib = round(int(current) / (1024**2), 2)
        logger.debug(f"cgroup memory.current: {current_mib} MiB")
    else:
        logger.debug("cgroup memory.current: unavailable")

    if swap and swap.isdigit():
        swap_mib = round(int(swap) / (1024**2), 2)
        logger.debug(f"cgroup memory.swap.current: {swap_mib} MiB")
    else:
        logger.debug("cgroup memory.swap.current: unavailable")


def memory_checkpoint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Контрольная точка управления оперативной памятью:
    снимает фрагментацию DataFrame, высвобождает ресурсы процесса
    при запуске с Docker контейнере с аллокатором glibc (ptmalloc2).

    Алгоритм работы:
    - Выводит статистику RAM через функцию rss_process_statistic.
    - Организует разрыв ссылок на старые версии DataFrame и Series через копирование
        со сменой имёни DataFrame, также это устраняет фрагментацию памяти между столбцами.
    - Выполняет сжатие кучи (heap) аллокатора в контейнере
      на Linux с glibc (ptmalloc2, задаётся Dockerfile, не меняется по ходу работы)
      вызывается malloc_trim(0): аллокатор glibc возвращает свободные страницы системе,
      что помогает снизить RSS процесса и избежать “залипаний” памяти.
    - Ещё раз выводит статистику RAM через функцию rss_process_statistic.

    Args:
        df (pd.DataFrame): Исходный DataFrame.
    Returns:
        pd.DataFrame: Копия DataFrame, готовая для дальнейшей работы pipeline.
    """

    logger.info("FUNCTION memory_checkpoint")
    logger.debug("Incoming statistics. Memory_checkpoint")
    # Проверим RSS процесса и объекты в RAM
    rss_process_statistic(df)
    # Проверим потребление памяти по cgroup
    cgroup_memory_statistic()

    # Выполняем паттерн разрыва связей
    df_new = df.copy()
    logger.debug("Copying completed")

    # Проверим RSS процесса и объекты в RAM
    rss_process_statistic(df_new)
    # Проверим потребление памяти по cgroup
    cgroup_memory_statistic()

    # Сжатие кучи на glibc (ptmalloc2).
    # Dockerfile определяет аллокатор — он не меняется динамически
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        logger.debug("malloc_trim not available (libc.so.6 missing)")
    else:
        logger.debug("malloc_trim(0) called successfully")

    logger.debug("Output statistics. Memory_checkpoint")

    # Проверим RSS процесса и объекты в RAM
    rss_process_statistic(df_new)
    # Проверим потребление памяти по cgroup
    cgroup_memory_statistic()

    return df_new


def heap_trim() -> None:
    """
    Контрольная точка управления памятью процесса под Linux/glibc:
    выполняет сжатие кучи (heap) аллокатора в контейнере
    на Linux с glibc (ptmalloc2, задаётся Dockerfile, не меняется по ходу работы)
    вызывается malloc_trim(0): аллокатор glibc возвращает свободные страницы системе,
    что помогает снизить RSS процесса и избежать “залипаний” памяти.

    Алгоритм работы:
    - Логирует текущее значение RSS процесса в MiB(мебибайты) для согласованности с
        выводом утилит мониторинга RAM.
    - Пытается загрузить libc.so.6 через ctypes и вызвать malloc_trim(0);
        при отсутствии libc фиксирует это в логе, при успехе логирует успешный вызов.
    - Повторно логирует RSS после операции для оценки возможного эффекта “trimming”
        в текущем окружении.
    Returns:
        None
    """

    logger.info("FUNCTION heap_trim")
    # Логируем общий RSS процесса
    logger.debug(f"RSS: {psutil.Process().memory_info().rss / 1024**2:.2f} MiB")

    # Проверим потребление памяти по cgroup
    cgroup_memory_statistic()

    # Сжатие кучи на glibc (ptmalloc2).
    # Dockerfile определяет аллокатор — он не меняется динамически
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        logger.debug("malloc_trim not available (libc.so.6 missing)")
    else:
        logger.debug("malloc_trim(0) called successfully")

    # Логируем общий RSS процесса
    logger.debug(f"RSS: {psutil.Process().memory_info().rss / 1024**2:.2f} MiB")

    # Проверим потребление памяти по cgroup
    cgroup_memory_statistic()
