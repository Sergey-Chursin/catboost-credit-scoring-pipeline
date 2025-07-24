import argparse
from log_config import setup_logging  # Переключатель уровня логирования

"""
Настраиваем парсер аргументов для CLI-запуска.
Это позволяет запускать скрипт с флагом --log-level INFO для вывода логов.
description - Описание скрипта для help-сообщения
"""
parser = argparse.ArgumentParser(description='Запуск пайплайна')
parser.add_argument('--log-level',
                    type=str,
                    default='OFF',  # По умолчанию логи отключены
                    help='Уровень логирования: INFO или OFF')

# Парсим аргументы из командной строки
args = parser.parse_args()

"""
Получаем логгер из импортированной функции.
Настраиваем логирование на основе аргумента.
'INFO' включит логи, любой другой или отсутствие аргумента — отключит.
"""
logger = setup_logging(args.log_level)




if __name__ == "__main__":
    # Пример: просто вывод, чтобы скрипт завершился
    print("Пайплайн завершён (тестовый запуск)")