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
    数据增强器：实现 Type II（非监督）数据增强，并应用论文中的误差过滤机制。
    - 生成模型同时生成特征和 target。
    - 使用一个基线模型（baseline_model）来预测生成特征，与生成 target 进行对比，
      根据预测误差（error < 0.03 * y_pred）筛选高质量数据。
    """

    def __init__(self, baseline_model):
        self.baseline_model = baseline_model  # 用于过滤的预训练基线模型
        self.generator = None
        self.metadata = None
        self.generator_method = None

    def fit_generator(
            self,
            full_df: pd.DataFrame,  # <--- 修改：现在 CTGAN 训练需要完整的 DataFrame (特征 + target)
            method: str = "CTGAN",
            epochs: int = 500,
            batch_size: int = 500,
            verbose: bool = True,
            **kwargs
    ):
        """在完整的 DataFrame (特征 + target) 上训练生成模型"""
        method_key = method.upper()
        build_kwargs = dict(kwargs)
        if method_key in {"CUSTOM_GAN", "CUSTOMGAN", "GAN"} and "verbose" not in build_kwargs:
            build_kwargs["verbose"] = verbose

        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(full_df)  # <--- detect from full_df

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
            print(f"开始训练 {method} 生成模型（epochs={epochs}）on full data...")

        start_time = time.time()
        self.generator.fit(full_df)  # <--- fit on full_df
        end_time = time.time()

        elapsed = end_time - start_time
        minutes, seconds = divmod(elapsed, 60)
        if verbose:
            print(f"生成模型训练完成！总耗时: {int(minutes)} 分 {seconds:.2f} 秒")

    def generate_and_predict(
            self,
            initial_samples: int = 10000,
            target_col: str = TARGET_COLUMN
    ) -> pd.DataFrame:
        """
        生成初始数据 (特征 + target) -> 应用论文后处理/过滤 (round, 负值置0, Feature_11约束) ->
        应用误差过滤 (generated target vs. baseline pred) -> 返回所有通过筛选的 syn_data
        """
        if self.generator is None:
            raise RuntimeError("请先调用 fit_generator 训练生成模型")

        print(f"{self.generator_method} 生成 {initial_samples} 条初始合成数据 (特征 + target)...")
        syn_raw_df = self.generator.sample(num_rows=initial_samples)

        # ===============================================
        # 1. 论文通用后处理 (round, 负值置0) - 作用于所有列
        # ===============================================
        print("应用论文通用后处理 (round, 负值置0)...")
        syn_raw_df = syn_raw_df.round(3)
        syn_raw_df[syn_raw_df < 0] = 0

        # ===============================================
        # 2. 论文特定特征逻辑约束 (Feature_11)
        # ===============================================
        if 'Feature_11' in syn_raw_df.columns:  # 检查列是否存在
            # 如果 Feature_11=0，则相关列置0
            syn_raw_df.loc[syn_raw_df['Feature_11'] == 0, ['Feature_12', 'Feature_13', 'Feature_14']] = 0
            # 排除无效组合：Feature_11 !=0 但相关列=0
            syn_raw_df = syn_raw_df[~((syn_raw_df['Feature_11'] != 0) &
                                      ((syn_raw_df['Feature_12'] == 0) | (syn_raw_df['Feature_13'] == 0) | (
                                                  syn_raw_df['Feature_14'] == 0)))]
            print("应用了 Feature_11 相关约束")
        else:
            print(f"警告: 列 '{'Feature_11'}' 不存在，跳过 Feature_11 相关约束。请检查列名是否与原始数据匹配。")

        # ===============================================
        # 3. 误差过滤 (生成 target vs. baseline 预测 target)
        #    这是你提供的代码片段中的核心过滤逻辑
        # ===============================================
        x_generated = syn_raw_df.drop(target_col, axis=1).values
        y_generated = syn_raw_df[target_col].values

        # 使用基线模型对生成特征进行预测
        y_pred_baseline = self.baseline_model.predict(x_generated)

        # 计算误差
        error = np.abs(y_generated - y_pred_baseline)

        # 筛选条件：误差小于 3% 的基线预测值
        initial_len = len(syn_raw_df)
        # 避免除以0，以及过滤掉 y_pred_baseline 是负值或0的情况
        # 确保 y_pred_baseline > 0 才能进行百分比误差计算
        valid_indices = (y_pred_baseline > 0) & (error < 0.03 * y_pred_baseline)

        syn_raw_df_filtered_error = syn_raw_df[valid_indices]

        print(f"误差过滤 (error < 0.03 * baseline_pred) 后保留 {len(syn_raw_df_filtered_error)} / {initial_len} 条")

        # ===============================================
        # 4. 负 target 过滤 (确保抗压强度 > 0)
        # ===============================================
        before_neg_filter = len(syn_raw_df_filtered_error)
        syn_data_final = syn_raw_df_filtered_error[syn_raw_df_filtered_error[target_col] > 0]
        print(f"负 target 过滤 ({target_col} > 0) 后保留 {len(syn_data_final)} / {before_neg_filter} 条")

        n_filtered = len(syn_data_final)
        print(f"最终过滤后剩余 {n_filtered} 条高质量合成数据 (总通过率: {n_filtered / initial_samples:.2%})")

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
        """使用 syn_data 池，按 ratios 合并数据并评估"""
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
        print(f"原数据量: {n_original} 条")

        # 如果未提供 syn_data，内部生成
        if syn_data is None:
            print("\n未提供 syn_data，内部生成 (特征 + target)...")
            syn_data = self.generate_and_predict(
                initial_samples=initial_samples,
                target_col=target_col
            )
        else:
            print(f"\n使用提供的 syn_data (大小: {len(syn_data)} 条)")

        # 循环每个 ratio，从 syn_data subsample
        for ratio in ratios:
            print(f"\n正在处理 ratio = {ratio} ...")

            if ratio == 0:
                aug_df = original_df.copy()
                n_synthetic_actual = 0
            else:
                n_target = int(ratio * n_original)
                if n_target > len(syn_data):
                    print(f"警告: 所需 {n_target} 条 > syn_data 池大小 {len(syn_data)}，使用所有")
                    syn_df = syn_data.copy()
                else:
                    syn_df = syn_data.sample(n=n_target, random_state=RANDOM_STATE)
                n_synthetic_actual = len(syn_df)
                aug_df = pd.concat([original_df, syn_df], ignore_index=True)

            # 分割
            X_aug = aug_df.drop(target_col, axis=1).values
            y_aug = aug_df[target_col].values
            x_train, x_test, y_train, y_test = train_test_split(
                X_aug, y_aug, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )

            # 重训 LightGBM
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
            print(f"Ratio {ratio}: RMSE = {rmse:.4f}, R² = {r2:.4f} (合成样本实际添加: {n_synthetic_actual})")

        return pd.DataFrame(results)
