import pandas as pd
import numpy as np
import time
from sdv.metadata import SingleTableMetadata
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from typing import Optional, Dict, List
from src.config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from src.model_train import compute_metrics
from src.synthesizers import build_synthesizer


class DataAugmentor:
    """
    Generate synthetic features and targets with baseline error filtering.

    Retain samples whose target error is below 3% of a positive baseline prediction.
    """

    def __init__(self, baseline_model):
        self.baseline_model = baseline_model  # Pretrained model used to filter synthetic samples.
        self.generator = None
        self.metadata = None
        self.generator_method = None

    def fit_generator(
            self,
            full_df: pd.DataFrame,
            method: str = "CTGAN",
            epochs: int = 500,
            batch_size: int = 500,
            verbose: bool = True,
            **kwargs
    ):
        """Train the synthesizer on a DataFrame containing features and the target."""
        method_key = method.upper()
        build_kwargs = dict(kwargs)
        if method_key in {"CUSTOM_GAN", "CUSTOMGAN", "GAN"} and "verbose" not in build_kwargs:
            build_kwargs["verbose"] = verbose

        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(full_df)

        self.generator = build_synthesizer(
            method,
            self.metadata,
            epochs=epochs,
            batch_size=batch_size,
            cuda=True,
            **build_kwargs
        )
        self.generator_method = method_key

        if verbose:
            print(f"Training {method} on the full dataset (epochs={epochs})...")

        start_time = time.time()
        self.generator.fit(full_df)
        end_time = time.time()

        elapsed = end_time - start_time
        minutes, seconds = divmod(elapsed, 60)
        if verbose:
            print(f"Synthesizer training completed in {int(minutes)} min {seconds:.2f} s")

    def generate_and_predict(
            self,
            initial_samples: int = 10000,
            target_col: str = TARGET_COLUMN
    ) -> pd.DataFrame:
        """
        Generate and return a filtered DataFrame of synthetic features and targets.

        Round values, clip negatives, enforce fiber constraints, and retain
        samples with positive targets that pass baseline prediction filtering.
        Fiber constraints support the complete Feature_11–Feature_14 column set
        or F_V (vol%), F_D, F_L, and F_M, with legacy names taking precedence.
        """
        if self.generator is None:
            raise RuntimeError("Call fit_generator before generating synthetic samples.")

        print(f"Generating {initial_samples} synthetic samples (features and target) with {self.generator_method}...")
        syn_raw_df = self.generator.sample(num_rows=initial_samples)

        # Round all generated columns and clip negative values to zero.
        print("Rounding generated values and clipping negatives to zero...")
        syn_raw_df = syn_raw_df.round(3)
        syn_raw_df[syn_raw_df < 0] = 0

        fiber_column_aliases = {
            "Feature_11": "F_V (vol%)",
            "Feature_12": "F_D",
            "Feature_13": "F_L",
            "Feature_14": "F_M",
        }
        fiber_columns = next(
            (columns for columns in (
                tuple(fiber_column_aliases), tuple(fiber_column_aliases.values())
            ) if all(column in syn_raw_df.columns for column in columns)),
            None,
        )

        # Enforce consistency between fiber volume and its dependent features.
        if fiber_columns is not None:
            volume_col, diameter_col, length_col, modulus_col = fiber_columns
            syn_raw_df.loc[syn_raw_df[volume_col] == 0, [diameter_col, length_col, modulus_col]] = 0
            # Exclude nonzero fiber volumes with zero-valued dependent features.
            syn_raw_df = syn_raw_df[~((syn_raw_df[volume_col] != 0) &
                                      ((syn_raw_df[diameter_col] == 0) | (syn_raw_df[length_col] == 0) | (
                                                  syn_raw_df[modulus_col] == 0)))]
            print(f"Applied fiber constraints using {volume_col}")
        else:
            print("Warning: No complete fiber column set found (Feature_11-Feature_14 or F_V (vol%), F_D, F_L, F_M); skipping fiber constraints.")

        # Filter synthetic samples using the baseline model.
        x_generated = syn_raw_df.drop(target_col, axis=1).values
        y_generated = syn_raw_df[target_col].values

        y_pred_baseline = self.baseline_model.predict(x_generated)

        error = np.abs(y_generated - y_pred_baseline)

        initial_len = len(syn_raw_df)
        # Require positive baseline predictions and less than 3% relative error.
        valid_indices = (y_pred_baseline > 0) & (error < 0.03 * y_pred_baseline)

        syn_raw_df_filtered_error = syn_raw_df[valid_indices]

        print(f"Baseline agreement filter (baseline_pred > 0, error < 0.03 * baseline_pred): retained {len(syn_raw_df_filtered_error)} / {initial_len} samples")

        # Retain strictly positive compressive strengths.
        before_neg_filter = len(syn_raw_df_filtered_error)
        syn_data_final = syn_raw_df_filtered_error[syn_raw_df_filtered_error[target_col] > 0]
        print(f"Positive target filter ({target_col} > 0): retained {len(syn_data_final)} / {before_neg_filter} samples")

        n_filtered = len(syn_data_final)
        print(f"Retained {n_filtered} synthetic samples after filtering (retention rate: {n_filtered / initial_samples:.2%})")

        return syn_data_final

    def run_augmentation_experiment(
            self,
            original_df: pd.DataFrame,
            ratios: List[float] = None,
            target_col: str = TARGET_COLUMN,
            lgb_params: Optional[Dict] = None,
            syn_data: Optional[pd.DataFrame] = None,
            initial_samples: int = 10000
    ) -> pd.DataFrame:
        """
        Evaluate augmentation ratios using a supplied or generated synthetic sample pool.

        Return a DataFrame of sample counts and test metrics for each ratio.
        """
        if ratios is None:
            ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        if lgb_params is None:
            lgb_params = {
                "objective": "regression",
                "metric": "rmse",
                "learning_rate": 0.05,
                "num_leaves": 128,
                "verbosity": -1
            }

        results = []
        n_original = len(original_df)
        print(f"Original dataset: {n_original} samples")

        if syn_data is None:
            print("\nNo synthetic pool supplied; generating features and targets...")
            syn_data = self.generate_and_predict(
                initial_samples=initial_samples,
                target_col=target_col
            )
        else:
            print(f"\nUsing the supplied synthetic pool ({len(syn_data)} samples)")

        # Sample synthetic rows in proportion to the original dataset size.
        for ratio in ratios:
            print(f"\nProcessing augmentation ratio = {ratio}...")

            if ratio == 0:
                aug_df = original_df.copy()
                n_synthetic_actual = 0
            else:
                n_target = int(ratio * n_original)
                if n_target > len(syn_data):
                    print(f"Warning: Requested {n_target} samples exceed the synthetic pool size ({len(syn_data)}); using all available samples")
                    syn_df = syn_data.copy()
                else:
                    syn_df = syn_data.sample(n=n_target, random_state=RANDOM_STATE)
                n_synthetic_actual = len(syn_df)
                aug_df = pd.concat([original_df, syn_df], ignore_index=True)

            X_aug = aug_df.drop(target_col, axis=1).values
            y_aug = aug_df[target_col].values
            x_train, x_test, y_train, y_test = train_test_split(
                X_aug, y_aug, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )

            # Train LightGBM on the augmented dataset.
            train_data = lgb.Dataset(x_train, label=y_train)
            valid_data = lgb.Dataset(x_test, label=y_test, reference=train_data)
            model = lgb.train(
                lgb_params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

            y_pred = model.predict(x_test)
            metrics = compute_metrics(y_test, y_pred)
            rmse = metrics["rmse"]
            r2 = metrics["r2"]

            results.append({
                "ratio": ratio,
                "n_synthetic": n_synthetic_actual,
                "total_samples": len(aug_df),
                "test_rmse": rmse,
                "test_r2": r2,
            })
            print(f"Ratio {ratio}: RMSE = {rmse:.4f}, R² = {r2:.4f} (synthetic samples added: {n_synthetic_actual})")

        return pd.DataFrame(results)
