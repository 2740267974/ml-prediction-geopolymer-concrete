"""Project-level configuration constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "origin_data.xlsx"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTION_FIGURES_DIR = FIGURES_DIR / "prediction"
AUGMENTATION_FIGURES_DIR = FIGURES_DIR / "augmentation"
SHAP_FIGURES_DIR = FIGURES_DIR / "shap"
SHAP_SCATTER_FIGURES_DIR = SHAP_FIGURES_DIR / "scatter"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"
RUNS_DIR = RESULTS_DIR / "runs"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FEATURES = 16
TARGET_COLUMN = "Com"
