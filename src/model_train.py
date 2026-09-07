from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.config import RANDOM_STATE


__all__ = [
    'train_lr',
    'train_decision_tree',
    'train_random_forest',
    'train_mlp',
    'train_xgboost',
    'train_lightgbm',
]

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the coefficient of determination (R²) from mean squared error
    and total sum of squares.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    y_mean = np.mean(y_true)
    tss = np.sum((y_true - y_mean) ** 2)
    mse = mean_squared_error(y_true, y_pred)

    if tss == 0:
        return float("nan")

    r2 = 1.0 - (len(y_true) * mse) / tss
    return float(r2)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics using flattened one-dimensional target arrays.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mape = mean_absolute_percentage_error(y_true, y_pred)

    r2 = compute_r2(y_true, y_pred)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
    }


@dataclass
class TrainResult:

    model: Any
    y_train_pred: np.ndarray
    y_test_pred: np.ndarray
    metrics: Dict[str, Dict[str, float]]

    params: Optional[Dict[str, Any]] = None
    # Stores the requested round-count argument passed to train_lightgbm.
    # LightGBM iteration settings in params can override this argument.
    num_boost_round: Optional[int] = None
    y_train_true: np.ndarray | None = None
    y_test_true: np.ndarray | None = None
    diagnostics: Optional[Dict[str, Any]] = None


def train_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 205,
) -> TrainResult:
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    if params is None:
        params = {
            "objective": "regression",
            "metric": {"l2", "l1"},
            "num_leaves": 69,
            "learning_rate": 0.32,
        }

    train_data = lgb.Dataset(x_train, label=y_train)
    test_data = lgb.Dataset(x_test, label=y_test, reference=train_data)

    model = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=[test_data],
        num_boost_round=num_boost_round,
    )

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    metrics = {"train": train_metrics, "test": test_metrics}

    print("=== Train Metrics ===")
    print(f"MAE : {train_metrics['mae']:.4f}")
    print(f"MSE : {train_metrics['mse']:.4f}")
    print(f"RMSE: {train_metrics['rmse']:.4f}")
    print(f"MAPE: {train_metrics['mape']:.4f}")
    print(f"R2  : {train_metrics['r2']:.4f}")

    print("\n=== Test Metrics ===")
    print(f"MAE : {test_metrics['mae']:.4f}")
    print(f"MSE : {test_metrics['mse']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAPE: {test_metrics['mape']:.4f}")
    print(f"R2  : {test_metrics['r2']:.4f}")

    return TrainResult(
        model=model,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        metrics=metrics,
        params=params,
        num_boost_round=num_boost_round,
    )

def train_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_estimators: int = 90,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
    max_depth: Optional[int] = None,
) -> TrainResult:
    y_train_1d = np.asarray(y_train).reshape(-1)
    y_test_1d = np.asarray(y_test).reshape(-1)

    params = {
        "n_estimators": n_estimators,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "max_depth": max_depth,
    }

    model = RandomForestRegressor(
        **params,
    )
    model.fit(x_train, y_train_1d)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_metrics = compute_metrics(y_train_1d, y_train_pred)
    test_metrics = compute_metrics(y_test_1d, y_test_pred)

    print("=== Train Metrics ===")
    print(f"MAE : {train_metrics['mae']:.4f}")
    print(f"MSE : {train_metrics['mse']:.4f}")
    print(f"RMSE: {train_metrics['rmse']:.4f}")
    print(f"MAPE: {train_metrics['mape']:.4f}")
    print(f"R2  : {train_metrics['r2']:.4f}\n")

    print("=== Test Metrics ===")
    print(f"MAE : {test_metrics['mae']:.4f}")
    print(f"MSE : {test_metrics['mse']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAPE: {test_metrics['mape']:.4f}")
    print(f"R2  : {test_metrics['r2']:.4f}")

    return TrainResult(
        model=model,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        metrics={"train": train_metrics, "test": test_metrics},
        params=params,
    )


def train_lr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    fit_intercept: bool = True,
    n_jobs: int | None = None,
) -> TrainResult:
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    model = LinearRegression(
        fit_intercept=fit_intercept,
        n_jobs=n_jobs,
    )

    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    return TrainResult(
        model=model,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        metrics={"train": train_metrics, "test": test_metrics},
        params={
            "fit_intercept": fit_intercept,
            "n_jobs": n_jobs,
        },
    )

def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> TrainResult:
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    model = MLPRegressor(
        hidden_layer_sizes=(85, 28),
        max_iter=10000,
        solver="adam",
        random_state=RANDOM_STATE,
        alpha=0.00001 * 3,
    )

    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    metrics = {
        "train": compute_metrics(y_train, y_train_pred),
        "test": compute_metrics(y_test, y_test_pred),
    }

    params = {
        "hidden_layer_sizes": (85, 28),
        "max_iter": 10000,
        "solver": "adam",
        "random_state": RANDOM_STATE,
        "alpha": 0.00001 * 3,
    }

    result = TrainResult(
        model=model,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        metrics=metrics,
        params=params,
        y_train_true=y_train,
        y_test_true=y_test,
    )
    return result


def train_decision_tree(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    random_state: int = RANDOM_STATE,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
) -> TrainResult:
    y_train_1d = np.asarray(y_train).reshape(-1)
    y_test_1d = np.asarray(y_test).reshape(-1)

    model = DecisionTreeRegressor(
        random_state=random_state,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )

    model.fit(x_train, y_train_1d)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_metrics = compute_metrics(y_train_1d, y_train_pred)
    test_metrics = compute_metrics(y_test_1d, y_test_pred)

    print("=== Train Metrics ===")
    print(f"MAE : {train_metrics['mae']:.4f}")
    print(f"MSE : {train_metrics['mse']:.4f}")
    print(f"RMSE: {train_metrics['rmse']:.4f}")
    print(f"MAPE: {train_metrics['mape']:.4f}")
    print(f"R2  : {train_metrics['r2']:.4f}\n")

    print("=== Test Metrics ===")
    print(f"MAE : {test_metrics['mae']:.4f}")
    print(f"MSE : {test_metrics['mse']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAPE: {test_metrics['mape']:.4f}")
    print(f"R2  : {test_metrics['r2']:.4f}")

    return TrainResult(
        model=model,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        metrics={"train": train_metrics, "test": test_metrics},
        params={
            "random_state": random_state,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        },
    )

def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    params: Optional[Dict[str, Any]] = None,
) -> TrainResult:
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    if params is None:
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 200,
            "learning_rate": 0.2,
            "max_depth": 6,
            "random_state": RANDOM_STATE,
        }

    model = XGBRegressor(**params)
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    metrics = {"train": train_metrics, "test": test_metrics}

    print("=== Train Metrics ===")
    print(f"MAE : {train_metrics['mae']:.4f}")
    print(f"MSE : {train_metrics['mse']:.4f}")
    print(f"RMSE: {train_metrics['rmse']:.4f}")
    print(f"MAPE: {train_metrics['mape']:.4f}")
    print(f"R2  : {train_metrics['r2']:.4f}")

    print("\n=== Test Metrics ===")
    print(f"MAE : {test_metrics['mae']:.4f}")
    print(f"MSE : {test_metrics['mse']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAPE: {test_metrics['mape']:.4f}")
    print(f"R2  : {test_metrics['r2']:.4f}")

    return TrainResult(
        model=model,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        metrics=metrics,
        params=params,
    )
