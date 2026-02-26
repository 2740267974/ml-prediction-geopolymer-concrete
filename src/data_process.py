# src/data_process.py
import os
import numpy as np
import pandas as pd


class DataHandler:
    """
    DataHandler is responsible for:
    1) Loading raw data from Excel / CSV files
    2) Converting all columns to numeric values
    3) Extracting features (X) and target (y) according to predefined rules

    This class follows the original preprocessing logic used in your experiments:
    - Features: first n_features columns
    - Target: last column
    """

    def __init__(self, data_path: str, n_features: int = 16):
        """
        Parameters
        ----------
        data_path : str
            Path to the dataset file (Excel or CSV).
        n_features : int, default=16
            Number of feature columns. The first n_features columns are treated as input features.
        """

        # Path to the raw dataset
        self.data_path = data_path

        # Number of feature columns
        self.n_features = n_features

        # Pandas DataFrame (useful for statistics and visualization)
        self.data_df = None

        # Numpy array version of the dataset
        self.data = None

        # Features (X) and target (y)
        self.a = None
        self.b = None

    def load_data(self):
        """
        Load and preprocess the dataset.

        Steps:
        1) Check whether the data file exists
        2) Read Excel or CSV file
        3) Convert all columns to numeric values (non-numeric values are set to NaN and filled with 0)
        4) Convert DataFrame to NumPy array
        5) Extract features and target

        Returns
        -------
        a : numpy.ndarray
            Feature matrix (X)
        b : numpy.ndarray
            Target vector (y)
        """

        # Step 0: Check file existence
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Step 1: Read data file
        ext = os.path.splitext(self.data_path)[1].lower()
        if ext in [".xlsx", ".xls"]:
            data = pd.read_excel(self.data_path)
        elif ext == ".csv":
            data = pd.read_csv(self.data_path)
        else:
            raise ValueError("Unsupported file format. Only .xlsx, .xls and .csv are supported.")

        # Step 2: Convert all columns to numeric and fill missing values with 0
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

        # Save DataFrame version for potential analysis or visualization
        self.data_df = data

        # Step 3: Convert DataFrame to NumPy array
        self.data = data.to_numpy(dtype=float)

        # Step 4: Extract features and target
        # Features: first n_features columns
        # Target: last column
        if self.data.shape[1] < self.n_features + 1:
            raise ValueError(
                f"Insufficient number of columns in the dataset. "
                f"Found {self.data.shape[1]} columns, but expected at least {self.n_features + 1} "
                f"(features + target)."
            )

        self.a = self.data[:, :self.n_features]
        self.b = self.data[:, -1]

        return self.a, self.b

    def get_features_and_target(self):
        """
        Return features and target after data loading.

        This method ensures that load_data() has been called before accessing the data.

        Returns
        -------
        a : numpy.ndarray
            Feature matrix (X)
        b : numpy.ndarray
            Target vector (y)
        """

        if self.a is None or self.b is None:
            raise RuntimeError("Data has not been loaded. Please call load_data() first.")

        return self.a, self.b
