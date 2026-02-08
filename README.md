# Machine Learning Prediction and SHAP Interpretability for Fiber-Reinforced Geopolymer Concrete

**Personal implementation** of a complete machine learning pipeline for predicting compressive strength and optimizing the life-cycle environmental impact of sustainable fiber-reinforced geopolymer concrete.

**Published Paper** (First-author contribution):  
M. Zhang, P. Guo, X. Tan, W. Meng, Y. Bao, "Cradle-to-gate assessment and optimization of sustainable geopolymer concrete", *Journal of Cleaner Production* (2026).  
DOI: [insert DOI when available, or write "In press"]

This repository contains **my independent implementation** of the following components:
- Literature-based data collection and preprocessing (>2,300 data points on composition, fiber, and curing parameters)
- Data augmentation using Conditional Tabular GAN (CTGAN)
- LightGBM regression modeling achieving excellent predictive performance (RMSE < 3.0 MPa)
- Comprehensive SHAP interpretability analysis (summary plots, dependence plots, force plots, and interaction analysis)
- Life Cycle Assessment (LCA) optimization to identify low-carbon mix designs

A small sample dataset is provided for demonstration and reproducibility. The full dataset is available in the paper's supplementary materials.

## Key Results
- High predictive accuracy: R² > 0.98, RMSE < 3.0 MPa on test set
- SHAP analysis quantitatively revealed the most influential parameters and their interactions
- LCA optimization identified environmentally superior geopolymer formulations

## Quick Start
```bash
pip install -r requirements.txt
jupyter lab notebooks/main_analysis.ipynb   # or jupyter notebook

