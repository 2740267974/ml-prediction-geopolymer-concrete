import json

import shap

def _is_tree_model(model) -> bool:

    name = model.__class__.__name__.lower()
    module = model.__class__.__module__.lower()

    if "xgboost" in module or "xgb" in name:
        # Prefer the fitted booster configuration over estimator parameters.
        try:
            booster = model.get_booster() if hasattr(model, "get_booster") else model
            config = json.loads(booster.save_config())
            booster_type = config["learner"]["gradient_booster"]["name"]
        except (AttributeError, KeyError, ValueError):
            booster_type = (
                model.get_xgb_params().get("booster")
                if hasattr(model, "get_xgb_params") else getattr(model, "booster", None)
            )
        return (booster_type or "gbtree") in {"gbtree", "dart"}

    if "lightgbm" in module or "lgbm" in name or "booster" in name:
        return True

    if "catboost" in module or "catboost" in name:
        return True

    if "sklearn" in module and any(k in name for k in [
        "randomforest", "gradientboosting", "extratrees",
        "decisiontree", "histgradientboosting"
    ]):
        return True

    # Detect other tree estimators by their prediction and tree attributes.
    if hasattr(model, "predict") and (hasattr(model, "trees_") or hasattr(model, "booster_")):
        return True

    return False  


def compute_shap_values(model, x_data, x_background=None, check_additivity: bool = False):
    """
    Compute SHAP values with automatic explainer selection.

    Use TreeExplainer for tree models and Explainer for other models.
    Pass non-callable estimators to Explainer through their predict method.

    Parameters
    ----------
    model : object
        Trained model to explain.
    x_data : pd.DataFrame or np.ndarray
        Feature data to explain.
    x_background : pd.DataFrame or np.ndarray or None
        Background data for non-tree explainers; defaults to x_data.
    check_additivity : bool
        Whether to check additivity when calling the tree explainer.

    Returns
    -------
    shap.Explanation
        SHAP values for x_data.
    """
    if _is_tree_model(model):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(x_data, check_additivity=check_additivity)
        return shap_values

    prediction_function = model if callable(model) else model.predict

    # Use the explained data as background when none is supplied.
    if x_background is None:
        explainer = shap.Explainer(prediction_function, x_data)
    else:
        explainer = shap.Explainer(prediction_function, x_background)

    shap_values = explainer(x_data)
    return shap_values
