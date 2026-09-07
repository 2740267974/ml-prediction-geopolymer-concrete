from pathlib import Path
from datetime import datetime

from src.config import PREDICTION_FIGURES_DIR, RESULTS_DIR

def get_results_dir(project_root: Path | None = None) -> Path:

    results_dir = RESULTS_DIR if project_root is None else project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def build_figure_path(model_name: str, filename: str | None = None) -> Path:

    model_dir = PREDICTION_FIGURES_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.png"

    return model_dir / filename
