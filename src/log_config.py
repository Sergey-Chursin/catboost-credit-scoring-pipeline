import logging

def setup_logging(level='OFF'):
    """
    Переключатель логирования функций пайплайна.
    При выборе опции 'info' будут выводиться названия функций,
    названия исходных обрабатываемых  признаков
    и названия новых фичей.
    В классификаторе будут выводиться названия методов,
    этапы обучения ансамбля и гиперпараметры моделей ансамбля.
    При выборе 'debug' дополнительно будут выводиться
    логи диагностики.
    При отсутствии ввода  логи отключаются.

    Args:
        level (str): 'info' для включения,
            любое другое для отключения.

            'debug' для детальной диагностики,
                    любое другое для отключения.
            По умолчанию 'OFF'.
    """

    # Сбрасываем старые настройки
    # Снимаем все блокировки, позволяя логам работать
    # на любом уровне (NOTSET — это "открытый" режим).
    logging.disable(logging.NOTSET)

    # Удаляем старые обработчики, чтобы избежать
    # конфликтов и basicConfig сработал
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Переводим аргумент в высокий регистр
    level_upper = level.upper()

    # Настраиваем вывод логов
    if level_upper == 'DEBUG':
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.getLogger().info("DEBUG logging mode enabled")

    elif level_upper == 'INFO':
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        logging.getLogger().info("INFO logging mode enabled")

    else:
        # Устанавливаем максимальный уровень логирования,
        # блокируя вывод уровня INFO и DEBUG
        logging.disable(logging.CRITICAL)

    return logger

# Создаём глобальный logger
logger = logging.getLogger(__name__)



