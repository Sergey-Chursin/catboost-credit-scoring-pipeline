# Используем официальный базовый образ с установленным Miniconda3
FROM continuumio/miniconda3

# Копируем файл зависимостей Conda (environment.yml) во временную папку внутри контейнера
COPY environment.yml /tmp/environment.yml

# Создаём новое Conda-окружение project_ml_env на основе environment.yml
RUN conda env create -f /tmp/environment.yml -n project_ml_env

# Добавляем команду активации окружения в ~/.bashrc для root,
# чтобы при каждом запуске bash автоматически активировалось нужное окружение
RUN echo "conda activate project_ml_env" >> /root/.bashrc

# Обновляем переменную среды PATH,
# чтобы бинарники окружения были доступны из любых команд
ENV PATH /opt/conda/envs/project_ml_env/bin:$PATH

# Устанавливаем рабочую директорию внутри контейнера — все команды и файлы будут относиться к /app
WORKDIR /app

# Копируем все файлы из вашей текущей папки на хосте в /app в контейнере
COPY . /app

# Открываем порт 8888 для доступа к Jupyter Notebook
EXPOSE 8888

# По умолчанию контейнер запускает интерактивную оболочку bash с уже активированным окружением
CMD ["bash"]
