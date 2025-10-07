import pandas as pd
import numpy as np
import pytest
from src.classifier import CatBoostEnsembleClassifier

# декораторо @pytest.fixture определяет фикстуру - делает результат функции доступным
# для использования в тестовых функциях
@pytest.fixture
def sample_data():
    """Сбалансированные синтетические данные для тестов (1000 сэмплов)."""
    # фиксируем "зерно" для воспроизводимости.
    np.random.seed(42)

    # по 500 объектов на класс
    n_per_class = 500
    X = pd.DataFrame({
        # Признак f1:
        # - первые 500 значений: нормальное распределение со средним 0
        # - вторые 500 значений: нормальное распределение, но сдвинутое на +1 по среднему
        "f1": np.concatenate(
            [np.random.randn(n_per_class),      # класс 0
             np.random.randn(n_per_class) + 1]  # класс 1
        ),
        # Признак f2:
        # - первые 500 значений: нормальное N(0, σ=5)
        # - вторые 500 значений: N(2, σ=5) — т.е. сдвинуто ещё и среднее на +2
        "f2": np.concatenate(
            [np.random.randn(n_per_class) * 5,      # класс 0
             np.random.randn(n_per_class) * 5 + 2]  # класс 1
        )
    })
    # Таргет
    # первые 500 объектов — класс 0,
    # следующие 500 — класс 1
    y = pd.Series([0] * n_per_class + [1] * n_per_class)

    return X, y



@pytest.fixture
def clf_small():
    """Классификатор с минимальными параметрами, чтобы тесты работали быстро."""
    return CatBoostEnsembleClassifier(
        # 5 фолдов + финальная
        params_list=[{'iterations': 1, 'verbose': 0}] * 6,
        weights_list=[1] * 6,
        n_splits=5
    )

def test_init_defaults():
    """Проверяем, что значения по умолчанию в __init__ корректно установлены."""
    clf = CatBoostEnsembleClassifier(
        params_list=[{'iterations': 1, 'verbose': 0}] * 6,
        weights_list=[1] * 6,
        n_splits=5
    )
    assert clf.threshold == 0.5      # порог классификации по умолчанию — 0.5
    assert clf.cat_features == []    # список категориальных фич пуст, если не задан
    assert clf.n_splits == 5          # количество фолдов равно переданному аргументу

def test_fit_creates_models(sample_data, clf_small):
    """
    Проверяем, что fit() создаёт нужное количество моделей
    и модели имеют методы предиктов.
    """
    X, y = sample_data
    clf_small.fit(X, y)
    # 5 фолдов + 1 финальная модель
    assert len(clf_small.models_) == 6
    # все модели имеют методы предиктов
    assert all(hasattr(m, "predict") for m in clf_small.models_)
    assert all(hasattr(m, "predict_proba") for m in clf_small.models_)

def test_fit_transform_returns_X(sample_data, clf_small):
    """Проверяем что fit_transform возвращает тот же X без изменений."""
    X, y = sample_data
    X_out = clf_small.fit_transform(X, y)
    # Результат должен быть равен исходному DataFrame
    assert X_out.equals(X)

def test_predict_proba_shape_and_sum(sample_data, clf_small):
    """Проверка корректности формы и нормировки метода predict_proba."""
    X, y = sample_data
    clf_small.fit(X, y)
    proba = clf_small.predict_proba(X)

    # результат — numpy массив
    assert isinstance(proba, np.ndarray)
    # две колонки: вероятности для классов 0 и 1
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(
        proba.sum(axis=1), 1.0, atol=1e-6     # сумма вероятностей по каждой строке ≈ 1
    )
    # np.testing.assert_allclose - проверяет, что два массива “почти равны”
    # не превышая погрешность atol=1e-6

def test_predict_output_binary_and_shape(sample_data, clf_small):
    """Проверка корректности формы и меток классов метода predict."""
    X, y = sample_data
    clf_small.fit(X, y)
    preds = clf_small.predict(X)

    # допустимы только классы 0 и 1
    assert set(np.unique(preds)) <= {0, 1}
    # длина массива предсказаний совпадает с числом объектов
    assert len(preds) == len(X)

def test_predict_with_custom_threshold(sample_data):
    """Проверяем влияние порога классификации на количество положительных срабатываний."""
    X, y = sample_data
    # Зададим очень низкий порог чтобы модели чаще предсказывали класс 1
    clf = CatBoostEnsembleClassifier(
        params_list=[{'iterations': 1, 'verbose': 0}] * 6,
        weights_list=[1] * 6,
        n_splits=5,
        threshold=0.1
    )
    clf.fit(X, y)
    preds_low = clf.predict(X)

    # Зададим высокий порог чтобы модели реже предсказывали класс 1
    clf.threshold = 0.9
    preds_high = clf.predict(X)

    # При низком пороге число 1 в предсказаниях должно быть больше или равно
    assert preds_low.sum() >= preds_high.sum()