from pydantic import BaseModel, Field


# Создаём схему запроса
# Определяем вложенную модель (структуру признаков одного объекта)
class FeatureVector(BaseModel):
    """
    Модель вектора признаков для одного кредитного объекта клиента.

    Бинаризовано - область значений поля разбивается на N непересекающихся промежутков,
        каждому промежутку случайным образом ставится в соответствие уникальный номер от 0 до N-1,
        значение поля заменяется номером промежутка, которому оно принадлежит.

    Закодировано - каждому уникальному значению поля случайным образом ставится в соответствие
        уникальный номер от 0 до K, значение поля заменяется номером этого значения.
    """

    id: int = Field(description="ID клиента")
    rn: int = Field(
        description="Порядковый номер кредитного продукта в кредитной истории"
    )
    pre_since_opened: int = Field(
        description="Дней с даты открытия кредита до даты сбора данных (бинаризовано)"
    )
    pre_since_confirmed: int = Field(
        description="Дней с даты подтверждения информации по кредиту до даты сбора данных (бинаризовано)"
    )
    pre_pterm: int = Field(
        description="Плановое количество дней с даты открытия кредита до даты закрытия (бинаризовано)"
    )
    pre_fterm: int = Field(
        description="Фактическое количество дней с даты открытия кредита до даты закрытия (бинаризовано)"
    )
    pre_till_pclose: int = Field(
        description="Плановое количество дней с даты сбора данных до даты закрытия кредита (бинаризовано)"
    )
    pre_till_fclose: int = Field(
        description="Фактическое количество дней с даты сбора данных до даты закрытия кредита (бинаризовано)"
    )
    pre_loans_credit_limit: int = Field(description="Кредитный лимит (бинаризовано)")
    pre_loans_next_pay_summ: int = Field(
        description="Сумма следующего платежа по кредиту (бинаризовано)"
    )
    pre_loans_outstanding: int = Field(
        description="Оставшаяся невыплаченная сумма кредита (бинаризовано)"
    )
    pre_loans_max_overdue_sum: int = Field(
        description="Максимальная просроченная задолженность (бинаризовано)"
    )
    pre_loans_credit_cost_rate: int = Field(
        description="Полная стоимость кредита (бинаризовано*"
    )
    pre_loans5: int = Field(description="Число просрочек до 5 дней (бинаризовано)")
    pre_loans530: int = Field(
        description="Число просрочек от 5 до 30 дней (бинаризовано)"
    )
    is_zero_loans5: int = Field(description="Флаг: нет просрочек до 5 дней")
    is_zero_loans530: int = Field(description="Флаг: нет просрочек от 5 до 30 дней")
    pre_util: int = Field(
        description="Отношение оставшейся невыплаченной суммы кредита к кредитному лимиту (бинаризовано)"
    )
    pre_over2limit: int = Field(
        description="Отношение текущей просроченной задолженности к кредитному лимиту (бинаризовано)"
    )
    is_zero_over2limit: int = Field(
        description="Флаг: отношение текущей просроченной задолженности к кредитному лимиту равняется 0"
    )
    enc_paym_0: int = Field(
        description="Статусы ежемесячных платежей за последние 0 месяцев (закодировано)"
    )
    enc_paym_1: int = Field(
        description="Статусы ежемесячных платежей за последние 1 месяцев (закодировано)"
    )
    enc_paym_2: int = Field(
        description="Статусы ежемесячных платежей за последние 2 месяцев (закодировано)"
    )
    enc_paym_8: int = Field(
        description="Статусы ежемесячных платежей за последние 8 месяцев (закодировано)"
    )
    enc_paym_9: int = Field(
        description="Статусы ежемесячных платежей за последние 9 месяцев (закодировано)"
    )
    enc_paym_10: int = Field(
        description="Статусы ежемесячных платежей за последние 10 месяцев (закодировано)"
    )
    enc_paym_24: int = Field(
        description="Статусы ежемесячных платежей за последние 24 месяцев (закодировано)"
    )
    enc_loans_account_holder_type: int = Field(
        description="Тип отношения к кредиту (закодировано)"
    )
    enc_loans_credit_status: int = Field(description="Статус кредита (закодировано)")
    enc_loans_credit_type: int = Field(description="Тип кредита (закодировано)")
    enc_loans_account_cur: int = Field(description="Валюта кредита (закодировано)")
    is_zero_loans3060: int = Field(description="Флаг: нет просрочек от 30 до 60 дней")
    is_zero_loans6090: int = Field(description="Флаг: нет просрочек от 60 до 90 дней")
    is_zero_loans90: int = Field(
        description="Флаг: нет просрочек более, чем на 90 дней"
    )
    enc_paym_3: int = Field(
        description="Статусы ежемесячных платежей за последние 3 месяцев (закодировано)"
    )
    enc_paym_4: int = Field(
        description="Статусы ежемесячных платежей за последние 3 месяцев (закодировано)"
    )
    enc_paym_5: int = Field(
        description="Статусы ежемесячных платежей за последние 5 месяцев (закодировано)"
    )
    enc_paym_6: int = Field(
        description="Статусы ежемесячных платежей за последние 6 месяцев (закодировано)"
    )
    enc_paym_7: int = Field(
        description="Статусы ежемесячных платежей за последние 7 месяцев (закодировано)"
    )
    enc_paym_11: int = Field(
        description="Статусы ежемесячных платежей за последние 11 месяцев (закодировано)"
    )
    enc_paym_12: int = Field(
        description="Статусы ежемесячных платежей за последние 12 месяцев (закодировано)"
    )
    enc_paym_13: int = Field(
        description="Статусы ежемесячных платежей за последние 13 месяцев (закодировано)"
    )
    enc_paym_14: int = Field(
        description="Статусы ежемесячных платежей за последние 14 месяцев (закодировано)"
    )
    enc_paym_15: int = Field(
        description="Статусы ежемесячных платежей за последние 15 месяцев (закодировано)"
    )
    enc_paym_16: int = Field(
        description="Статусы ежемесячных платежей за последние 16 месяцев (закодировано)"
    )
    enc_paym_17: int = Field(
        description="Статусы ежемесячных платежей за последние 17 месяцев (закодировано)"
    )
    enc_paym_18: int = Field(
        description="Статусы ежемесячных платежей за последние 18 месяцев (закодировано)"
    )
    enc_paym_19: int = Field(
        description="Статусы ежемесячных платежей за последние 19 месяцев (закодировано)"
    )
    enc_paym_20: int = Field(
        description="Статусы ежемесячных платежей за последние 20 месяцев (закодировано)"
    )
    enc_paym_21: int = Field(
        description="Статусы ежемесячных платежей за последние 21 месяцев (закодировано)"
    )
    enc_paym_22: int = Field(
        description="Статусы ежемесячных платежей за последние 22 месяцев (закодировано)"
    )
    enc_paym_23: int = Field(
        description="Статусы ежемесячных платежей за последние 23 месяцев (закодировано)"
    )


# Второй моделью делаем запрос – список объектов FeatureVector
# Запрос - это словарь с одним ключом 'data' и списком словарей внутри
class PredictionRequest(BaseModel):
    """Модель запроса на получение предсказаний для одного или нескольких клиентов."""

    data: list[FeatureVector]


class SinglePrediction(BaseModel):
    """Модель предсказания по одному клиенту."""

    client_id: int = Field(description="ID клиента")
    probability: float = Field(description="Вероятность дефолта (0.0 - 1.0)")
    class_label: int = Field(description="Бинарная метка (0 = не дефолт, 1 = дефолт)")


# Модель ответа - список словарей(1 словарь = 1 клиент) и информация о модели
class PredictionResponse(BaseModel):
    """Модель ответа с предсказаниями для всех клиентов."""

    predictions: list[SinglePrediction]
    model_info: str = "CatBoost ensemble, 6 models, ROC AUC = 0.7558"
