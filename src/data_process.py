from pathlib import Path

import numpy as np
import pandas as pd

from src.config import N_FEATURES, TARGET_COLUMN

class DataHandler:

    def __init__(
        self,
        data_path: str | Path,
        n_features: int = N_FEATURES,
        target_column: str = TARGET_COLUMN,
    ):

        self.data_path = Path(data_path)

        self.n_features = n_features
        self.expected_target_column = target_column

        self.data_df: pd.DataFrame | None = None
        self.data: np.ndarray | None = None

        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

        # Backward-compatible aliases used by existing notebooks.
        self.a: np.ndarray | None = None
        self.b: np.ndarray | None = None

        self.feature_names: list[str] = []
        self.target_name: str | None = None
        self.n_missing_or_non_numeric_filled: int = 0

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        ext = self.data_path.suffix.lower()
        if ext in [".xlsx", ".xls"]:
            data = pd.read_excel(self.data_path)
        elif ext == ".csv":
            data = pd.read_csv(self.data_path)
        else:
            raise ValueError("Unsupported file format. Only .xlsx, .xls and .csv are supported.")

        self._validate_columns(data)

        numeric_data = data.apply(pd.to_numeric, errors="coerce")
        self.n_missing_or_non_numeric_filled = int(numeric_data.isna().sum().sum())
        numeric_data = numeric_data.fillna(0)

        self.data_df = numeric_data
        self.data = numeric_data.to_numpy(dtype=float)

        self.feature_names = list(numeric_data.columns[: self.n_features])
        self.target_name = str(numeric_data.columns[-1])

        self.X = self.data[:, : self.n_features]
        self.y = self.data[:, -1]

        self.a = self.X
        self.b = self.y

        return self.X, self.y

    def get_features_and_target(self) -> tuple[np.ndarray, np.ndarray]:

        if self.X is None or self.y is None:
            raise RuntimeError("Data has not been loaded. Please call load_data() first.")

        return self.X, self.y

    def get_summary(self) -> dict[str, object]:
        """
        Return a compact summary of the loaded dataset.
        """
        if self.data_df is None or self.X is None or self.y is None:
            raise RuntimeError("Data has not been loaded. Please call load_data() first.")

        return {
            "data_path": str(self.data_path),
            "n_samples": int(self.X.shape[0]),
            "n_features": int(self.X.shape[1]),
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "n_missing_or_non_numeric_filled": self.n_missing_or_non_numeric_filled,
        }

    def _validate_columns(self, data: pd.DataFrame) -> None:
        """
        Validate that the dataset has enough columns and the expected target.
        """
        expected_min_cols = self.n_features + 1
        if data.shape[1] < expected_min_cols:
            raise ValueError(
                f"Insufficient number of columns in the dataset. "
                f"Found {data.shape[1]} columns, but expected at least {expected_min_cols} "
                f"(features + target)."
            )

        actual_target = str(data.columns[-1])
        if self.expected_target_column and actual_target != self.expected_target_column:
            raise ValueError(
                f"Unexpected target column. Expected last column '{self.expected_target_column}', "
                f"but found '{actual_target}'."
            )
        
