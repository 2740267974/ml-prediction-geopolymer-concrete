"""
Minimal standalone LightGBM prediction example.

The complete research workflow, including generative data augmentation
and SHAP analysis, is provided in `main_pipeline.ipynb`.
"""

from pathlib import Path
import sys

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_process import DataHandler
from src.config import DATA_PATH, FIGURES_DIR, N_FEATURES, RANDOM_STATE, TEST_SIZE
from src.model_train import train_lightgbm
from src.plotter import DataPlotter

def main() -> None:
    output_path = FIGURES_DIR / "LightGBM.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_handler = DataHandler(str(DATA_PATH), n_features=N_FEATURES)
    data_handler.load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        data_handler.a,
        data_handler.b,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    result = train_lightgbm(x_train, y_train, x_test, y_test)

    y_train_true = result.y_train_true if result.y_train_true is not None else y_train
    y_test_true = result.y_test_true if result.y_test_true is not None else y_test

    plotter = DataPlotter()
    fig, _ = plotter.plot(
        y_train_true,
        result.y_train_pred,
        y_test_true,
        result.y_test_pred,
        result.metrics["train"]["r2"],
        result.metrics["test"]["r2"],
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved prediction plot to {output_path}")

if __name__ == "__main__":
    main()
