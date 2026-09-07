import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap

class DataPlotter:
    """Create regression, augmentation, and SHAP figures with optional saving."""

    def __init__(self):
        plt.rcParams["font.family"] = "Times New Roman"

    def plot(
        self,
        y_train: np.ndarray,
        y_train_pred: np.ndarray,
        y_test: np.ndarray,
        y_test_pred: np.ndarray,
        r2_train: float,
        r2_test: float,
    ):
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(
            y_train, y_train_pred,
            edgecolor="blue", facecolors="none",
            marker="o", s=25,
        )

        ax.scatter(
            y_test, y_test_pred,
            edgecolor="red", facecolors="none",
            marker="o", s=25,
        )

        max_val = max(
            float(np.max(y_train)), float(np.max(y_train_pred)),
            float(np.max(y_test)), float(np.max(y_test_pred)),
        )
        ax.set_xlim(0, max_val + 5)
        ax.set_ylim(0, max_val + 5)

        # Reference line for perfect predictions.
        ax.plot([0, max_val + 5], [0, max_val + 5], "k--", lw=1)

        ax.text(max_val * 0.16, max_val * 0.95, "Zero error", fontsize=24)
        ax.text(max_val * 0.16, max_val * 0.87, f"Training set (R² = {r2_train:.2f})", fontsize=24)
        ax.text(max_val * 0.16, max_val * 0.79, f"Testing set (R² = {r2_test:.2f})", fontsize=24)

        # Draw markers alongside the metric annotations.
        ax.plot([max_val * 0.03, max_val * 0.13], [max_val * 0.965, max_val * 0.965], "k--", lw=1)
        ax.scatter(max_val * 0.08, max_val * 0.885, edgecolor="red", facecolors="none", marker="o", s=25)
        ax.scatter(max_val * 0.08, max_val * 0.805, edgecolor="blue", facecolors="none", marker="o", s=25)

        ax.tick_params(labelsize=24)
        ax.set_xlabel("Actual compressive strength (MPa)", fontsize=24)
        ax.set_ylabel("Predicted compressive strength (MPa)", fontsize=24)

        return fig, ax

    def plot_augmentation_results(self, results_df: pd.DataFrame, ylabel: str = "RMSE (MPa)",
                                  save_path: str = None):
        """
        Plot RMSE against augmentation ratio for each model and return (fig, ax).

        Require Augmentation_ratio, Model, and RMSE columns in results_df.
        Convert ratios to percentages in place and optionally save to save_path.
        """
        results_df['RMSE'] = pd.to_numeric(results_df['RMSE'], errors='coerce')
        results_df['Augmentation_ratio'] = results_df['Augmentation_ratio'] * 100  # Express augmentation ratios as percentages.

        colors = {
            'CTGAN': '#f781bf',
            'TVAE': '#ff7f0e',
            'CTABGAN': '#1f77b4',
        }

        y_min = results_df['RMSE'].min() - 0.05
        y_max = results_df['RMSE'].max() + 0.05

        fig, ax = plt.subplots(figsize=(10, 6))

        for model in results_df['Model'].unique():
            subset = results_df[results_df['Model'] == model]
            ax.scatter(subset['Augmentation_ratio'], subset['RMSE'],
                       label=model, color=colors.get(model, '#1f77b4'), s=60, alpha=0.8)

        # Highlight the baseline without augmentation.
        zero_subset = results_df[results_df['Augmentation_ratio'] == 0]
        if not zero_subset.empty:
            baseline_rmse = zero_subset['RMSE'].iloc[0]
            ax.scatter(0, baseline_rmse, color='black', s=60, alpha=0.8)

        # Annotate the lowest RMSE and its augmentation ratio.
        min_rmse = results_df['RMSE'].min()
        min_row = results_df.loc[results_df['RMSE'].idxmin()]
        min_ratio = min_row['Augmentation_ratio']

        ax.axhline(y=min_rmse, color='gray', linestyle='--', linewidth=1)
        ax.text(min_ratio + 2, min_rmse - 0.01,
                f'RMSE ({min_ratio:.0f}%) = {min_rmse:.3f}',
                color='gray', fontsize=28)

        ax.text(2, baseline_rmse - 0.01,
                f'RMSE (0%) = {baseline_rmse:.3f}',
                color='gray', fontsize=28)

        ax.set_xlabel('Augmentation ratio (%)', fontsize=28)
        ax.set_ylabel(ylabel, fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.set_xlim(-5, 105)
        ax.set_ylim(y_min, y_max)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3,
                  frameon=False, prop={'family': 'Times New Roman', 'size': 28},
                  columnspacing=0.5, labelspacing=0.2, handletextpad=0.3, borderaxespad=0.2)

        plt.subplots_adjust(top=0.9, bottom=0.1)
        plt.grid(False)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)

        return fig, ax

    def plot_aug_results_rmse(
            self,
            aug_results: pd.DataFrame,
            ratio_col: str = "ratio",
            rmse_col: str = "test_rmse",
            ylabel: str = "RMSE (MPa)",
            label: str = "Augmentation",
            save_path: str = None
    ):
        """
        Plot a single model's augmentation results and return (fig, ax).

        Read fractional ratios and RMSE from ratio_col and rmse_col in aug_results.
        Optionally save the figure to save_path.
        """
        df = aug_results.copy()
        df[ratio_col] = pd.to_numeric(df[ratio_col], errors="coerce")
        df[rmse_col] = pd.to_numeric(df[rmse_col], errors="coerce")
        df = df.dropna(subset=[ratio_col, rmse_col]).sort_values(ratio_col)

        # Express augmentation ratios as percentages.
        x = (df[ratio_col].values * 100.0)
        y = df[rmse_col].values

        baseline_val = None
        baseline_rows = df[df[ratio_col] == 0]
        if not baseline_rows.empty:
            baseline_val = float(baseline_rows[rmse_col].iloc[0])

        min_idx = int(np.nanargmin(y))
        min_x = float(x[min_idx])
        min_y = float(y[min_idx])

        y_min = float(np.nanmin(y)) - 0.05
        y_max = float(np.nanmax(y)) + 0.05

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(x, y, s=60, alpha=0.8, color="#f781bf", label=label)

        # Highlight the baseline without augmentation.
        if baseline_val is not None:
            ax.scatter([0], [baseline_val], s=60, alpha=0.8, color="black")
            ax.text(2, baseline_val - 0.01, f"RMSE (0%) = {baseline_val:.3f}",
                    color="gray", fontsize=28)

        # Annotate the lowest RMSE and its augmentation ratio.
        ax.axhline(y=min_y, color="gray", linestyle="--", linewidth=1)
        ax.text(min_x + 2, min_y - 0.01, f"RMSE ({min_x:.0f}%) = {min_y:.3f}",
                color="gray", fontsize=28)

        ax.set_xlabel("Augmentation ratio (%)", fontsize=28)
        ax.set_ylabel(ylabel, fontsize=28)
        ax.tick_params(axis="both", which="major", labelsize=28)

        ax.set_xlim(-5, 105)
        ax.set_ylim(y_min, y_max+2)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1),
            ncol=1,
            frameon=False,
            prop={"family": "Times New Roman", "size": 28},
            columnspacing=0.5,
            labelspacing=0.2,
            handletextpad=0.3,
            borderaxespad=0.2
        )

        plt.subplots_adjust(top=0.9, bottom=0.1)
        ax.grid(False)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)

        return fig, ax


    def plot_shap_summary(
        self,
        shap_values,
        x_data: pd.DataFrame,
        feature_names=None,
        ylabel: str = "SHAP value (impact on model output)",
        save_path: str = None
    ):
        """
        Plot a SHAP beeswarm summary and return (fig, ax).

        Use shap_values and x_data with optional feature_names for custom labels.
        The ylabel argument sets the horizontal axis label. Optionally save the
        figure to save_path.
        """

        if feature_names is None:
            if isinstance(x_data, pd.DataFrame):
                feature_names = list(x_data.columns)
            else:
                raise ValueError("feature_names must be provided if x_data is not a DataFrame.")

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            x_data,
            feature_names=feature_names,
            show=False
        )

        fig = plt.gcf()
        ax = plt.gca()

        ax.set_xlabel(ylabel, fontsize=24)
        ax.tick_params(axis="both", which="major", labelsize=20)

        # The final axes contains the SHAP colorbar.
        if len(fig.axes) >= 2:
            cbar_ax = fig.axes[-1]
            cbar_ax.set_ylabel("Feature value", fontsize=20)
            cbar_ax.tick_params(labelsize=18)

        plt.subplots_adjust(left=0.25)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)

        return fig, ax


    def plot_shap_scatter_all(
        self,
        shap_values,
        x_data: pd.DataFrame,
        feature_names=None,
        save_dir: str = None,
        filename_prefix: str = "feature",
        point_size: int = 50,
    ):
        """
        Create SHAP scatter plots for all features and return saved file paths.

        Save figures when save_dir is supplied and close each figure after plotting.
        """

        if feature_names is None:
            if isinstance(x_data, pd.DataFrame):
                feature_names = list(x_data.columns)
            else:
                n_features = shap_values.values.shape[1]
                feature_names = [f"f{i}" for i in range(n_features)]

        saved = []
        for i, name in enumerate(feature_names):
            save_path = None
            if save_dir is not None:
                import os
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{filename_prefix}_{i}_{name}.png")

            fig, ax = self.plot_shap_scatter_one(
                shap_values=shap_values,
                x_data=x_data,
                feature=i,
                feature_names=feature_names,
                point_size=point_size,
                save_path=save_path
            )
            plt.close(fig)

            if save_path is not None:
                saved.append(save_path)

        return saved
    def plot_shap_scatter_one(
        self,
        shap_values,
        x_data: pd.DataFrame,
        feature: int | str,
        feature_names=None,
        xlabel: str = None,
        ylabel: str = None,
        point_size: int = 50,
        remove_colorbar: bool = True,
        save_path: str = None,
    ):
        """
        Plot SHAP values for one feature and return (fig, ax).

        Select the feature by index or name from a shap.Explanation. Infer feature
        names from x_data when it is a DataFrame; otherwise require feature_names.
        Optionally save the figure to save_path.
        """

        import shap
        import matplotlib

        if feature_names is None:
            if isinstance(x_data, pd.DataFrame):
                feature_names = list(x_data.columns)
            else:
                raise ValueError("x_data must be a DataFrame or you must provide feature_names.")

        if isinstance(feature, str):
            if feature not in feature_names:
                raise ValueError(f"Feature '{feature}' not found in feature_names.")
            feature_index = feature_names.index(feature)
            feature_name = feature
        else:
            feature_index = int(feature)
            feature_name = feature_names[feature_index]

        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        shap_values_for_feature = shap_values[:, feature_index]

        shap.plots.scatter(
            shap_values_for_feature,
            color=shap_values.data[:, feature_index],
            hist=False,
            ax=ax,
            show=False
        )

        for collection in ax.collections:
            if isinstance(collection, matplotlib.collections.PathCollection):
                collection.set_sizes([point_size])

        if remove_colorbar:
            for collection in getattr(ax, "collections", []):
                if hasattr(collection, "colorbar") and collection.colorbar is not None:
                    collection.colorbar.remove()

        ax.axhline(0, color="black", linewidth=1, linestyle="--")

        for s in ["top", "right", "bottom", "left"]:
            ax.spines[s].set_visible(True)
            ax.spines[s].set_linewidth(2)

        ax.tick_params(axis="both", which="major", labelsize=30)

        if xlabel is None:
            xlabel = feature_name
        if ylabel is None:
            ylabel = f"SHAP value for {feature_name}"

        ax.set_xlabel(xlabel, fontsize=36)
        ax.set_ylabel(ylabel, fontsize=36)

        fig = plt.gcf()

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax

