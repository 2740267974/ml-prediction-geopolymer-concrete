
# src/model_train.py
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

__all__ = [
    'train_lr',
    'train_decision_tree',
    'train_random_forest',
    'train_mlp',
    'train_xgboost',
    'train_lightgbm',
    # 如果以后还加了其他train_函数，也补到这里
]

def r2_like_yours(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R^2 using the same style as your original code.

    Notes
    -----
    Your original implementation was:
        R2 = 1 - (len(y) * MSE) / TSS
    where:
        TSS = sum((y - mean(y))^2)
    This is mathematically equivalent to the standard R^2 for MSE defined by sklearn.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    y_mean = np.mean(y_true)
    tss = np.sum((y_true - y_mean) ** 2)
    mse = mean_squared_error(y_true, y_pred)

    # Avoid division by zero if y_true is constant (rare but possible)
    if tss == 0:
        return float("nan")

    r2 = 1.0 - (len(y_true) * mse) / tss
    return float(r2)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics in a unified and robust way.

    - Ensures y_true and y_pred are 1D arrays
    - Uses your original R² definition (r2_like_yours)
    - Compatible with all models (LightGBM / RF / SVR / ANN / GAN)
    """
    # ✅ Ensure correct shape
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # ✅ Keep YOUR R² definition
    r2 = r2_like_yours(y_true, y_pred)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
    }


@dataclass
class TrainResult:
    """
    Unified training result container for all models.

    Mandatory fields (used by plotter & notebook):
    - model
    - y_train_pred
    - y_test_pred
    - metrics

    Optional fields (model-specific):
    - params
    - num_boost_round
    """

    # ===== mandatory (ALL models) =====
    model: Any
    y_train_pred: np.ndarray
    y_test_pred: np.ndarray
    metrics: Dict[str, Dict[str, float]]

    # ===== optional (model-specific) =====
    params: Optional[Dict[str, Any]] = None
    num_boost_round: Optional[int] = None
    y_train_true: np.ndarray | None = None
    y_test_true: np.ndarray | None = None


def train_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 205,
) -> TrainResult:
    """
    Train a LightGBM regression model and evaluate it on train/test sets.

    Parameters
    ----------
    x_train, y_train : numpy arrays
        Training features and target.
    x_test, y_test : numpy arrays
        Testing features and target.
    params : dict, optional
        LightGBM parameters. If None, a default set (similar to your original) is used.
    num_boost_round : int
        Number of boosting iterations (your original was 205).

    Returns
    -------
    TrainResult
        Contains model, predictions, and metrics for train/test.
    """

    # 1) Default parameters (close to your original code)
    if params is None:
        params = {
            "objective": "regression",
            "metric": {"l2", "l1"},
            "num_leaves": 69,
            "learning_rate": 0.32,
        }

    # 2) Build LightGBM datasets
    train_data = lgb.Dataset(x_train, label=y_train)
    test_data = lgb.Dataset(x_test, label=y_test, reference=train_data)

    # 3) Train model
    model = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=[test_data],
        num_boost_round=num_boost_round,
    )

    # 4) Predict on train/test
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # 5) Compute metrics
    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    metrics = {"train": train_metrics, "test": test_metrics}

    # 6) Print metrics (same spirit as your original script)
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
    random_state: int = 8,
    n_jobs: int = -1,
    max_depth: Optional[int] = None,
) -> TrainResult:
    """
    Train a RandomForest regressor and return predictions + metrics.

    This follows the same "TrainResult" format as train_lightgbm,
    so your notebook can swap models easily.
    """
    # Make sure y is 1D (your old code used y_train.ravel())
    y_train_1d = np.asarray(y_train).reshape(-1)
    y_test_1d = np.asarray(y_test).reshape(-1)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        max_depth=max_depth,
    )
    model.fit(x_train, y_train_1d)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_metrics = compute_metrics(y_train_1d, y_train_pred)
    test_metrics = compute_metrics(y_test_1d, y_test_pred)

    # (optional) print like your LightGBM function does
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
    """
    Train a Linear Regression model (sklearn) and compute metrics.

    Parameters
    ----------
    x_train, y_train, x_test, y_test:
        Train/test split data.
    fit_intercept:
        Whether to fit intercept term (default True).
    n_jobs:
        Parallel jobs for sklearn LinearRegression (may be ignored depending on sklearn version).

    Returns
    -------
    TrainResult
        Same format as LightGBM training result.
    """

    # ✅ Make sure y is 1D
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    # 1) Model init
    model = LinearRegression(
        fit_intercept=fit_intercept,
        n_jobs=n_jobs,
    )

    # 2) Fit
    model.fit(x_train, y_train)

    # 3) Predict
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # 4) Metrics (统一格式)
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
) -> tuple[TrainResult, np.ndarray, np.ndarray]:
    """
    Train MLP model with prediction filtering.

    NOTE:
    - This function RETURNS filtered y_train / y_test
    - So notebook must use returned y_train_f / y_test_f for plotting
    """

    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    # 1. Define model (same as your original code)
    model = MLPRegressor(
        hidden_layer_sizes=(85, 28),
        max_iter=10000,
        solver="adam",
        random_state=43,
        alpha=0.00001 * 3,
    )

    # 2. Train
    model.fit(x_train, y_train)

    # 3. Predict
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # 4. Filter predictions (MLP-specific logic)
    train_mask = (y_train_pred <= 150) & (y_train_pred > 0)
    test_mask = (y_test_pred <= 150) & (y_test_pred > 0)

    y_train_f = y_train[train_mask]
    y_test_f = y_test[test_mask]

    y_train_pred_f = y_train_pred[train_mask]
    y_test_pred_f = y_test_pred[test_mask]

    # 5. Metrics (统一格式)
    metrics = {
        "train": compute_metrics(y_train_f, y_train_pred_f),
        "test": compute_metrics(y_test_f, y_test_pred_f),
    }

    params = {
        "hidden_layer_sizes": (85, 28),
        "max_iter": 10000,
        "solver": "adam",
        "random_state": 43,
        "alpha": 0.00001 * 3,
        "filter_pred_range": "(0, 200]",
    }

    result = TrainResult(
        model=model,
        y_train_pred=y_train_pred_f,
        y_test_pred=y_test_pred_f,
        metrics=metrics,
        params=params,
        y_train_true=y_train_f,  # ✅ 过滤后的真实值
        y_test_true=y_test_f,  # ✅ 过滤后的真实值
    )
    return result


def train_decision_tree(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    random_state: int = 184,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
) -> TrainResult:
    """
    Train a DecisionTreeRegressor and return TrainResult (same format as LightGBM).

    This is compatible with your plotter usage:
        result.metrics["train"]["r2"]
        result.y_train_pred
        result.y_test_pred
    """

    # ✅ keep y as 1D (same as your old y_train.ravel())
    y_train_1d = np.asarray(y_train).reshape(-1)
    y_test_1d = np.asarray(y_test).reshape(-1)

    # 1) Init model (matches your original clf = DecisionTreeRegressor(random_state=184))
    model = DecisionTreeRegressor(
        random_state=random_state,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )

    # 2) Fit
    model.fit(x_train, y_train_1d)

    # 3) Predict
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # 4) Metrics (use your unified compute_metrics + r2_like_yours)
    train_metrics = compute_metrics(y_train_1d, y_train_pred)
    test_metrics = compute_metrics(y_test_1d, y_test_pred)

    # (optional) print like other models
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
    """
    Train an XGBoost regressor and return TrainResult (same format as LightGBM/RF/LR/DT/MLP).

    Parameters
    ----------
    x_train, y_train, x_test, y_test:
        Train/test split data (already split outside).
    params:
        Optional dict for XGBRegressor hyperparameters.
        If None, we use your original settings.

    Returns
    -------
    TrainResult
        model, predictions, metrics, and params
    """

    # ✅ Make sure y is 1D (your old code used y_train.ravel())
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    # 1) Default hyperparameters (same as your original script)
    if params is None:
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 200,
            "learning_rate": 0.2,
            "max_depth": 6,
            # (optional but recommended for stability)
            # "subsample": 1.0,
            # "colsample_bytree": 1.0,
            # "random_state": 417,
        }

    # 2) Init + train
    model = XGBRegressor(**params)
    model.fit(x_train, y_train)

    # 3) Predict
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # 4) Metrics (统一格式)
    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    metrics = {"train": train_metrics, "test": test_metrics}

    # 5) Print metrics (optional, same style as LightGBM)
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

