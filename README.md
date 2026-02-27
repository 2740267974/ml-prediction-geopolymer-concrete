# Machine Learning Prediction and SHAP Interpretability for Fiber-Reinforced Geopolymer Concrete

Personal implementation of a machine learning pipeline for predicting the compressive strength of fiber-reinforced geopolymer concrete, including tabular data augmentation (CTGAN / TVAE) and SHAP interpretability analysis.

---

## Related Paper

M. Zhang, P. Guo, X. Tan, W. Meng, Y. Bao,  
*Cradle-to-gate assessment and optimization of sustainable geopolymer concrete*,  
**Journal of Cleaner Production**, 2026, 538: 147387.  

DOI: https://doi.org/10.1016/j.jclepro.2025.147387  
Link: https://www.sciencedirect.com/science/article/pii/S0959652625027441

> Note:  
> This repository focuses on the machine learning prediction, data augmentation, and SHAP interpretability implementation.  
> The Life Cycle Assessment (LCA) optimization part from the paper is not included in this codebase.

---

## Project Overview

This repository includes:

- Data preprocessing and train/test split
- Baseline regression modeling (LightGBM / XGBoost depending on experiment)
- Tabular data augmentation:
  - CTGAN (Conditional Tabular GAN)
  - TVAE (Tabular Variational Autoencoder)
- SHAP interpretability analysis:
  - SHAP summary plot (beeswarm)
  - SHAP feature-wise scatter plots
- Structured result saving:
  - Figures → `results/figures/`
  - Tables → `results/tables/`

A small sample dataset is included for demonstration purposes.  
The full dataset is available via the published paper and supplementary materials.

---


# Quick Start Guide

This document explains how to run the project correctly.


## 1. Install Dependencies

Make sure you are in the project root directory:

``` bash
pip install -r requirements.txt
```


## 2. Run the Notebooks

### Demo (Recommended First)

``` bash
jupyter notebook notebooks/demo_quick.ipynb
```


### Full Research Pipeline

``` bash
jupyter notebook notebooks/research_full.ipynb
```


## 3. Expected Outputs

After running the notebooks, the following files will be generated:

``` text
results/figures/baseline_pred.png
results/figures/augmentation_rmse.png
results/figures/shap_summary.png
results/figures/shap_scatter/
results/tables/augmentation_experiment.csv
```


## Notes

-   Always start Jupyter from the project root directory.
-   The demo notebook runs quickly and is recommended for first-time
    users.
-   The full research notebook includes data augmentation and SHAP
    analysis.
