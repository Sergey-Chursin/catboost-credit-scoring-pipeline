import logging


def setup_logging(level: str = "OFF") -> logging.Logger:
    """
    Переключатель уровней логирования.

    Args:
        level (str): Задаёт режим логирования.
        'info' - выводятся основные логи.
        'debug' - выводятся детальные логи диагностики памяти.
         По умолчанию 'OFF' - логи не выводятся.
    Returns:
        logging.Logger: Логгер для текущего модуля.
        Внимание: настройка применяется глобально для всей подсистемы logging в проекте.
    """

    # Сбрасываем старые настройки.
    # Снимаем все блокировки - разрешаем вывод всех логов
    # (NOTSET — это "открытый" режим).
    logging.disable(logging.NOTSET)

    # Удаляем старые обработчики, чтобы избежать
    # конфликтов и basicConfig смог примениться
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Переводим аргумент в высокий регистр
    level_upper = level.upper()

    # Настраиваем вывод логов
    if level_upper == "DEBUG":
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.getLogger().info("DEBUG logging mode enabled")

    elif level_upper == "INFO":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
        logging.getLogger().info("INFO logging mode enabled")

    else:
        # Устанавливаем максимальный уровень логирования,
        # блокируя вывод уровня INFO и DEBUG
        logging.disable(logging.CRITICAL)

    return logger


# Создаём глобальный logger
logger = logging.getLogger(__name__)
