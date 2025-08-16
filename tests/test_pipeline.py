import pytest

from unittest.mock import patch, MagicMock
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from pipeline import (
    main_pipeline,
    load_pipeline,
    run_train_coordinator,
    run_test_coordinator,
    run_inference_coordinator
)

# ------- Тест для main_pipeline -------------
def test_main_pipeline_returns_pipeline():
    """
    Проверяет, что main_pipeline возвращает объект sklearn.pipeline.Pipeline,
    и что в пайплайне есть требуемые шаги - 'preprocessing' и 'classifier'.
    """
    pipeline = main_pipeline(
        sample_frac=0.5,
        params_list=[{"param": 1}],
        weights_list=[1.0],
        threshold=0.5,
        cat_features=['cat1'],
        n_splits=2,
        seed=42,
        shuffle=True,
        prop_features_dict={"f": 1},
        mean_freq_source_list=["mean_f"],
        drop_list=["to_drop"],
        logger=None
    )

    # Проверяем, что pipeline - объект класса Pipeline (sklearn)
    assert isinstance(pipeline, Pipeline)

    # Получаем имена шагов pipeline и убеждаемся, что нужные шаги присутствуют
    step_names = dict(pipeline.steps).keys()
    assert 'preprocessing' in step_names
    assert 'classifier' in step_names


# ------- Тест для load_pipeline -------------
def test_load_pipeline_success(tmp_path):
    """
    Проверяет успешную загрузку объекта пайплайна из файла.
    """
    obj = {'some': 'pipeline'}
    path = tmp_path / "pipeline.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)

    loaded = load_pipeline(str(path))
    # Проверяем, что содержимое файла успешно загружено
    assert loaded == obj


def test_load_pipeline_not_found():
    """
    Проверяет, что при попытке загрузить отсутствующий пайплайн
    возникает FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        load_pipeline("bad_path/fake.pkl")


# ------- Тест для run_train_coordinator -------------

# @patch - при запуске теста все функции в pipeline
# будут заменены на объекты MagicMock
# Декораторы “оборачивают” функцию поочерёдно, начиная с самой глубокой (ближайшей к функции).
# Самый внешний декоратор добавляет аргумент первым.
# Самый внутренний декоратор добавляет аргумент последним.
# Каждый декоратор, когда оборачивает функцию, создаёт новую функцию с добавленным аргументом.
# Следующий декоратор уже “видит” функцию с одним аргументом и добавляет свой аргумент ещё одним “слоем”.
# Поэтому самый последний @patch (стоящий ближе к функции) получает первое место среди аргументов теста.
# Самый верхний @patch добавит аргумент уже к функции, которая и так уже имеет все прежние аргументы.
@patch("pipeline.main_pipeline")
@patch("pipeline.split_dataset_by_target")
@patch("pipeline.load_dataset")
@patch("pipeline.check_data_folder_and_count_files")
@patch("pipeline.pickle.dump")
def test_run_train_coordinator_full_cycle(
        mock_pickle_dump,
        mock_check_count,
        mock_load_dataset,
        mock_split_dataset,
        mock_main_pipeline,
        tmp_path
):
    """
    Проверяет, что run_train_coordinator корректно проходит все этапы
    при работе с мок-данными: вызывает main_pipeline, осуществляет fit,
    и результат обучения сохраняется с помощью pickle.dump.
    """
    # Моки для загрузки и деления исходных данных
    mock_check_count.return_value = ("some_folder", 1)
    mock_load_dataset.return_value = "raw_data"
    mock_split_dataset.return_value = {
        "X_train": np.zeros((2, 2)),
        "y_train": np.array([0, 1]),
        "X_test": np.zeros((1, 2)),
        "y_test": np.array([1])
    }
    # Мокаем pipeline и fit
    # Создаём заглушку
    pipe_mock = MagicMock()
    # Задаём ей поведение для метода fit - вернуть себя же
    pipe_mock.fit.return_value = pipe_mock
    # Передаём её в результат mock который заменяет ф-ю main_pipeline
    mock_main_pipeline.return_value = pipe_mock

    out_path = tmp_path / "model.pkl"
    run_train_coordinator(
        pipeline_path=str(out_path),
        raw_data_path="raw",
        temp_data_path="temp",
        pre_features=["feature_1"],
        num_parts_to_preprocess_at_once=1,
        pattern="*",
        target_path="target.csv",
        train_size=0.8,
        seed_split_dataset=11,
        stratify_col="target",
        sample_frac=0.5,
        params_list=[{"param": 1}],
        weights_list=[1.0],
        threshold=0.4,
        cat_features=["cat"],
        n_splits=2,
        seed=1,
        shuffle=True,
        eval_metric="off",
        verbose=False,
        prop_features_dict={"feature": 1},
        mean_freq_source_list=["f"],
        drop_list=["d"],
        classes_metric_list=["acc"],
        logger=None
    )
    # Проверяем что вызывался main_pipeline
    # called - вернёт True если метод был вызван
    assert mock_main_pipeline.called
    # Проверяем, что у pipe_mock вызывался метод fit с тренировочными данными
    assert pipe_mock.fit.called
    # Проверяем, что результат был сохранён через pickle.dump
    assert mock_pickle_dump.called


# ------- Тест для run_test_coordinator -------------
@patch("pipeline.load_pipeline")
@patch("pipeline.check_data_folder_and_count_files")
@patch("pipeline.load_dataset")
@patch("pipeline.split_dataset_by_target")
@patch("pipeline.make_file_path")
@patch("pipeline.save_predictions_with_id")
@patch("pipeline.compute_and_log_metrics")
def test_run_test_coordinator_flow(
        # не используется в тесте, но нужен для run_test_coordinator
        mock_compute_metrics,
        mock_save,
        mock_make_file_path,
        mock_split,
        mock_load,
        mock_check,
        mock_load_pipe
):
    """
    Проверяет, что run_test_coordinator проходит все этапы с моками:
    - пайплайн загружается,
    - данные разделяются,
    - модель делает предсказания,
    - результат сохраняется.
    """
    mock_pipe = MagicMock()
    mock_pipe.predict_proba.return_value = np.array([[0.1, 0.9]])
    mock_pipe.predict.return_value = np.array([1])
    mock_load_pipe.return_value = mock_pipe
    mock_check.return_value = ("some_folder", 1)
    mock_load.return_value = {"dataset": 1}
    mock_split.return_value = {
        "X_train": {"id": pd.Series([1, 2, 3])},
        "y_train": np.array([0, 1, 0]),
        "X_test": {"id": pd.Series([3, 4])},
        "y_test": np.array([0, 1])
    }
    mock_make_file_path.return_value = "predictions.csv"

    run_test_coordinator(
        pipeline_path="path.pkl",
        raw_data_path="raw",
        temp_data_path="temp",
        pre_features=["a"],
        num_parts_to_preprocess_at_once=1,
        pattern="*",
        target_path="target.csv",
        train_size=0.8,
        seed_split_dataset=42,
        stratify_col="some_cat",
        test_predict_path=".",
        predict_file_extension=".csv",
        output="proba",
        eval_metrics="acc",
        classes_metric_list=["acc"],
        verbose=False,
        logger=None
    )
    # Проверяем что пайплайн загружается с правильным путём
    mock_load_pipe.assert_called_once_with("path.pkl")
    # Проверяем что вызывается split_dataset_by_target
    assert mock_split.called
    # Проверяем, что был вызван predict_proba
    assert mock_pipe.predict_proba.called
    # Проверяем, что результат был сохранён функцией save_predictions_with_id
    assert mock_save.called


# ------- Тест для run_inference_coordinator -------------
@patch("pipeline.load_pipeline")
@patch("pipeline.check_data_folder_and_count_files")
@patch("pipeline.load_dataset")
@patch("pipeline.make_file_path")
@patch("pipeline.save_predictions_with_id")
def test_run_inference_coordinator_flow(
        mock_save,
        mock_make_file_path,
        mock_load_dataset,
        mock_check,
        mock_load_pipe
):
    """
    Проверяет, что run_inference_coordinator проходит полный процесс получения
    и сохранения предсказаний на новых данных:
    - пайплайн загружается,
    - данные грузятся,
    - производится предсказание,
    - результат сохраняется.
    """
    mock_pipe = MagicMock()
    mock_pipe.predict_proba.return_value = np.array([[0.2, 0.8]])
    mock_pipe.predict.return_value = np.array([1])
    mock_load_pipe.return_value = mock_pipe
    mock_check.return_value = ("any_folder", 1)
    # Мокаем возвращаемые данные как словари из Series
    mock_load_dataset.return_value = {"id": pd.Series([1, 2]), "f": pd.Series([100, 200])}
    mock_make_file_path.return_value = "predictions.csv"

    run_inference_coordinator(
        pipeline_path="path.pkl",
        data_path="raw",
        temp_data_path="temp",
        pre_features=["feature"],
        num_parts_to_preprocess_at_once=1,
        pattern="*",
        predict_file_extension=".csv",
        output="proba",
        output_dir="any_folder",
        verbose=False,
        logger=None
    )
    # Проверяем, что пайплайн действительно загружался с нужным путём
    mock_load_pipe.assert_called_once_with("path.pkl")
    # Проверяем что папка проверялась
    assert mock_check.called
    # Проверяем, что данные считались
    assert mock_load_dataset.called
    # Проверяем, что pipeline вызывал predict_proba
    assert mock_pipe.predict_proba.called
    # Проверяем, что результат предсказания был сохранён
    assert mock_save.called