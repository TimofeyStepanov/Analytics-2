from datetime import datetime

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold
from sklearn.preprocessing import StandardScaler


# код с пары
def train_first(df_for_model):
    df = df_for_model.dropna()

    # Определение целевой переменной и признаков
    df = df.drop(columns=['Unnamed: 0'])
    target = 'raw_mix.lab.measure.sito_009'
    features = df.columns[df.columns != target]

    # Разделение данных
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Скалирование данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Построение модели
    model = LinearRegression()

    model.fit(X_train_scaled, y_train)

    # Оценка модели
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mae_baseline = mean_absolute_error(y_test, np.full_like(y_test, y_test.mean()))
    mae_baseline_shift = mean_absolute_error(y_test[:-1], y_test[1:])

    # Вычисление разностей для реальных и предсказанных значений
    y_test_diff = y_test[1:].values - y_test[:-1].values
    y_pred_diff = y_pred[1:] - y_pred[:-1]
    # Подсчет доли сонаправленных изменений
    same_direction = np.sum((y_test_diff * y_pred_diff) > 0) / len(y_test_diff)

    print(f'Доля сонаправленных изменений: {same_direction:.3f}')
    print(f'Mean Absolute Error Baseline: {mae_baseline}')
    print(f'Mean Absolute Error Baseline shifted: {mae_baseline_shift}')
    print(f'Mean Absolute Error: {mae}')

    return {
        'features': list(features),
        'metrics': {
            'mae': float(mae),
            'mae_baseline': float(mae_baseline),
            'mae_baseline_shift': float(mae_baseline_shift),
            'same_direction_ratio': float(same_direction),
            'mse': float(mse)
        },
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }


# мой код
def train_second(df_for_model, n_splits=10):
    # Подготовка данных
    df = df_for_model.dropna()
    target = 'raw_mix.lab.measure.sito_009'
    features = df.columns[df.columns != target].tolist()

    # Инициализация кросс-валидации
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mae_scores, mse_scores, direction_ratios = [], [], []

    for train_index, test_index in tscv.split(df):
        # Разбиение на train/test
        X_train, X_test = df[features].iloc[train_index], df[features].iloc[test_index]
        y_train, y_test = df[target].iloc[train_index], df[target].iloc[test_index]

        # Скалирование
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # jбучение модели
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)

        # Предсказание и метрики
        y_pred = model.predict(X_test_scaled)
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))

        # Сонаправленные изменения
        y_test_diff = y_test[1:].values - y_test[:-1].values
        y_pred_diff = y_pred[1:] - y_pred[:-1]
        direction_ratios.append(np.mean((y_test_diff * y_pred_diff) > 0))

    # Усреднение метрик по фолдам
    avg_mae = np.mean(mae_scores)
    avg_mse = np.mean(mse_scores)
    avg_direction = np.mean(direction_ratios)

    # Baseline-метрики (считаем на последнем фолде для примера)
    last_baseline = mean_absolute_error(y_test, np.full_like(y_test, y_test.mean()))
    last_baseline_shift = mean_absolute_error(y_test[:-1], y_test[1:])

    print(f'Доля сонаправленных изменений (средняя): {avg_direction:.3f}')
    # print(f'Mean Squared Error (средний): {avg_mse:.4f}')
    print(f'MAE Baseline (последний фолд): {last_baseline:.4f}')
    print(f'MAE Baseline Shift (последний фолд): {last_baseline_shift:.4f}')
    print(f'Mean Absolute Error (средний): {avg_mae:.4f}')

    return {
        'features': features,
        'metrics': {
            'mae': float(avg_mae),
            'mse': float(avg_mse),
            'same_direction_ratio': float(avg_direction),
            'mae_baseline': float(last_baseline),
            'mae_baseline_shift': float(last_baseline_shift)
        },
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }


def train_third(df_for_model, n_splits=10):
    # Подготовка данных
    df = df_for_model.dropna()
    target = 'raw_mix.lab.measure.sito_009'
    features = df.columns[df.columns != target].tolist()

    # bнициализация кросс-валидации
    tscv = KFold(n_splits=n_splits)
    mae_scores, mse_scores, direction_ratios = [], [], []

    for train_index, test_index in tscv.split(df):
        # разбиение на train/test
        X_train, X_test = df[features].iloc[train_index], df[features].iloc[test_index]
        y_train, y_test = df[target].iloc[train_index], df[target].iloc[test_index]

        # скалирование
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Обучение модели
        model = CatBoostRegressor(
            iterations=200,
            depth=4,
            l2_leaf_reg=5,
            silent=True
        )
        model.fit(X_train_scaled, y_train)

        # предсказание и метрики
        y_pred = model.predict(X_test_scaled)
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))

        # сонаправленные изменения
        y_test_diff = y_test[1:].values - y_test[:-1].values
        y_pred_diff = y_pred[1:] - y_pred[:-1]
        direction_ratios.append(np.mean((y_test_diff * y_pred_diff) > 0))

    # усреднение метрик
    avg_mae = np.mean(mae_scores)
    avg_mse = np.mean(mse_scores)
    avg_direction = np.mean(direction_ratios)

    # baseline-метрики
    last_baseline = mean_absolute_error(y_test, np.full_like(y_test, y_test.mean()))
    last_baseline_shift = mean_absolute_error(y_test[:-1], y_test[1:])

    print(f'Доля сонаправленных изменений (средняя): {avg_direction:.3f}')
    # print(f'Mean Squared Error (средний): {avg_mse:.4f}')
    print(f'MAE Baseline (последний фолд): {last_baseline:.4f}')
    print(f'MAE Baseline Shift (последний фолд): {last_baseline_shift:.4f}')
    print(f'Mean Absolute Error (средний): {avg_mae:.4f}')

    return {
        'features': features,
        'metrics': {
            'mae': float(avg_mae),
            'mse': float(avg_mse),
            'same_direction_ratio': float(avg_direction),
            'mae_baseline': float(last_baseline),
            'mae_baseline_shift': float(last_baseline_shift)
        },
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }


if __name__ == "__main__":
    # Загрузка данных
    df = pd.read_csv('./data/processed/mart.csv')

    train_first(df)
    print('\n')
    train_second(df)
    print('\n')
    train_third(df)
