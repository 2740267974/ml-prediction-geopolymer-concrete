# src/shap_analysis.py
import shap

def _is_tree_model(model) -> bool:

    name = model.__class__.__name__.lower()
    module = model.__class__.__module__.lower()

    # LightGBM native booster
    if "lightgbm" in module or "lgbm" in name or "booster" in name:
        return True

    # XGBoost
    if "xgboost" in module or "xgb" in name:
        return True

    # CatBoost
    if "catboost" in module or "catboost" in name:
        return True

    # sklearn tree models
    if "sklearn" in module and any(k in name for k in [
        "randomforest", "gradientboosting", "extratrees",
        "decisiontree", "histgradientboosting"
    ]):
        return True

    # fallback: objects with predict + tree structure often work with TreeExplainer
    if hasattr(model, "predict") and (hasattr(model, "trees_") or hasattr(model, "booster_")):
        return True

    return False  


def compute_shap_values(model, x_data, x_background=None, check_additivity: bool = False):
    """
    Compute SHAP values with automatic explainer selection.

    - If model is tree-based: use shap.TreeExplainer (fast & stable).
    - Otherwise: use shap.Explainer (generic).

    Parameters
    ----------
    model : trained model
    x_data : pd.DataFrame or np.ndarray
        data to explain
    x_background : pd.DataFrame or np.ndarray or None
        background data for non-tree explainers (optional)
    check_additivity : bool
        passed to tree explainer call

    Returns
    -------
    shap.Explanation
    """
    if _is_tree_model(model):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(x_data, check_additivity=check_additivity)
        return shap_values

    # non-tree: generic explainer (may be slower)
    if x_background is None:
        explainer = shap.Explainer(model, x_data)
    else:
        explainer = shap.Explainer(model, x_background)

    shap_values = explainer(x_data)
    return shap_values
