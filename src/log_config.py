import logging

def setup_logging(level='OFF'):
    """
    Переключатель логирования функций пайплайна.
    При выборе опции 'INFO' будут выводиться названия функций,
    названия исходных обрабатываемых  признаков
    и названия новых фичей.
    В классификаторе будут выводиться названия методов,
    этапы обучения ансамбля и гиперпараметры моделей ансамбля.
    При отсутствии ввода или ошибке ввода логи отключаются.

    Args:
        level (str): 'INFO' для включения,
            любое другое для отключения.
            По умолчанию 'OFF'.
    """

    # При правильном вводе
    if level.upper() == 'INFO':
        # Снимаем все блокировки, позволяя логам работать
        # на любом уровне (NOTSET — это "открытый" режим).
        logging.disable(logging.NOTSET)

        # Удаляем старые обработчики, чтобы избежать
        # конфликтов и basicConfig сработал
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # Настраиваем вывод логов
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        # Выводим сообщение
        logging.getLogger().info("INFO logging mode")

    else:
        # Снимаем все блокировки, позволяя логам работать
        # на любом уровне (NOTSET — это "открытый" режим).
        logging.disable(logging.NOTSET)

        # Удаляем старые обработчики, чтобы избежать конфликтов
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Устанавливаем максимальный уровень логирования,
        # блокируя вывод уровня INFO и ниже
        logging.disable(logging.CRITICAL)

    return logger

# Создаём глобальный logger
logger = logging.getLogger(__name__)