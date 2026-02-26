# src/plotter.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class DataPlotter:
    """
    Plot only (no saving, no showing).
    Return (fig, ax) so notebook decides show/save.
    """

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
        # ✅ CHANGED: create figure explicitly and RETURN it
        fig, ax = plt.subplots(figsize=(8, 8))

        # Training set (blue hollow circles)
        ax.scatter(
            y_train, y_train_pred,
            edgecolor="blue", facecolors="none",
            marker="o", s=25,
        )

        # Testing set (red hollow circles)
        ax.scatter(
            y_test, y_test_pred,
            edgecolor="red", facecolors="none",
            marker="o", s=25,
        )

        # Axis limits
        max_val = max(
            float(np.max(y_train)), float(np.max(y_train_pred)),
            float(np.max(y_test)), float(np.max(y_test_pred)),
        )
        ax.set_xlim(0, max_val + 5)
        ax.set_ylim(0, max_val + 5)

        # y = x line
        ax.plot([0, max_val + 5], [0, max_val + 5], "k--", lw=1)

        # Text
        ax.text(max_val * 0.16, max_val * 0.95, "Zero error", fontsize=24)
        ax.text(max_val * 0.16, max_val * 0.87, f"Training set (R² = {r2_train:.2f})", fontsize=24)
        ax.text(max_val * 0.16, max_val * 0.79, f"Testing set (R² = {r2_test:.2f})", fontsize=24)

        # Small legend-style markers
        ax.plot([max_val * 0.03, max_val * 0.13], [max_val * 0.965, max_val * 0.965], "k--", lw=1)
        ax.scatter(max_val * 0.08, max_val * 0.885, edgecolor="red", facecolors="none", marker="o", s=25)
        ax.scatter(max_val * 0.08, max_val * 0.805, edgecolor="blue", facecolors="none", marker="o", s=25)

        # Labels
        ax.tick_params(labelsize=24)
        ax.set_xlabel("Actual compressive strength (MPa)", fontsize=24)
        ax.set_ylabel("Predicted compressive strength (MPa)", fontsize=24)

        # ✅ CHANGED: return fig for saving
        return fig, ax

    def plot_augmentation_results(self, results_df: pd.DataFrame, ylabel: str = "RMSE (eV/atom)",
                                  save_path: str = None):
        """
        绘制生成模型数据增强效果图（augmentation ratio vs RMSE）
        results_df 必须包含列：'Augmentation_ratio', 'Model', 'RMSE'
        """
        import pandas as pd
        import matplotlib.pyplot as plt

        # 确保数值类型
        results_df['RMSE'] = pd.to_numeric(results_df['RMSE'], errors='coerce')
        results_df['Augmentation_ratio'] = results_df['Augmentation_ratio'] * 100  # 转为百分比

        # 颜色定义
        colors = {
            'CTGAN': '#f781bf',  # 粉色
            'TVAE': '#ff7f0e',  # 橙色（可扩展）
            'CTABGAN': '#1f77b4',  # 蓝色（备用）
        }

        # y轴范围
        y_min = results_df['RMSE'].min() - 0.05
        y_max = results_df['RMSE'].max() + 0.05

        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制每个模型的散点
        for model in results_df['Model'].unique():
            subset = results_df[results_df['Model'] == model]
            ax.scatter(subset['Augmentation_ratio'], subset['RMSE'],
                       label=model, color=colors.get(model, '#1f77b4'), s=60, alpha=0.8)

        # 绘制 ratio=0 的黑色点（baseline）
        zero_subset = results_df[results_df['Augmentation_ratio'] == 0]
        if not zero_subset.empty:
            baseline_rmse = zero_subset['RMSE'].iloc[0]
            ax.scatter(0, baseline_rmse, color='black', s=60, alpha=0.8)

        # 最小RMSE标注
        min_rmse = results_df['RMSE'].min()
        min_row = results_df.loc[results_df['RMSE'].idxmin()]
        min_ratio = min_row['Augmentation_ratio']

        ax.axhline(y=min_rmse, color='gray', linestyle='--', linewidth=1)
        ax.text(min_ratio + 2, min_rmse - 0.01,
                f'RMSE ({min_ratio:.0f}%) = {min_rmse:.3f}',
                color='gray', fontsize=28)

        # baseline标注
        ax.text(2, baseline_rmse - 0.01,
                f'RMSE (0%) = {baseline_rmse:.3f}',
                color='gray', fontsize=28)

        # 坐标轴设置
        ax.set_xlabel('Augmentation ratio (%)', fontsize=28)
        ax.set_ylabel(ylabel, fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.set_xlim(-5, 105)
        ax.set_ylim(y_min, y_max)

        # 图例
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
            save_path: str = None
    ):
        """
        画 run_augmentation_experiment 输出的 aug_results（单模型）:
        需要包含列：ratio (0~1) 和 test_rmse
        不show，只返回(fig, ax)，save_path可选
        """
        df = aug_results.copy()
        df[ratio_col] = pd.to_numeric(df[ratio_col], errors="coerce")
        df[rmse_col] = pd.to_numeric(df[rmse_col], errors="coerce")
        df = df.dropna(subset=[ratio_col, rmse_col]).sort_values(ratio_col)

        # ratio -> percent
        x = (df[ratio_col].values * 100.0)
        y = df[rmse_col].values

        # baseline (0%)
        baseline_val = None
        baseline_rows = df[df[ratio_col] == 0]
        if not baseline_rows.empty:
            baseline_val = float(baseline_rows[rmse_col].iloc[0])

        # min point
        min_idx = int(np.nanargmin(y))
        min_x = float(x[min_idx])
        min_y = float(y[min_idx])

        y_min = float(np.nanmin(y)) - 0.05
        y_max = float(np.nanmax(y)) + 0.05

        fig, ax = plt.subplots(figsize=(10, 6))

        # main scatter (pink like CTGAN in your example)
        ax.scatter(x, y, s=60, alpha=0.8, color="#f781bf", label="CTGAN")

        # baseline black point
        if baseline_val is not None:
            ax.scatter([0], [baseline_val], s=60, alpha=0.8, color="black")
            ax.text(2, baseline_val - 0.01, f"RMSE (0%) = {baseline_val:.3f}",
                    color="gray", fontsize=28)

        # min rmse line + text
        ax.axhline(y=min_y, color="gray", linestyle="--", linewidth=1)
        ax.text(min_x + 2, min_y - 0.01, f"RMSE ({min_x:.0f}%) = {min_y:.3f}",
                color="gray", fontsize=28)

        ax.set_xlabel("Augmentation ratio (%)", fontsize=28)
        ax.set_ylabel(ylabel, fontsize=28)
        ax.tick_params(axis="both", which="major", labelsize=28)

        ax.set_xlim(-5, 105)
        ax.set_ylim(y_min, y_max+2)

        # legend (single model)
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