# src/save_utils.py
from pathlib import Path
from datetime import datetime


def get_results_dir(project_root: Path | None = None) -> Path:
    """
    Return the results directory path.
    If not exists, create it automatically.
    """
    if project_root is None:
        # Assume this file is src/save_utils.py
        project_root = Path(__file__).resolve().parents[1]

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def build_figure_path(model_name: str, filename: str | None = None) -> Path:
    """
    Build a save path for figures.

    Example:
    results/LightGBM/LightGBM_20260208_233046.png
    """
    results_dir = get_results_dir()
    model_dir = results_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.png"

    return model_dir / filename
