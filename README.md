# Machine Learning and GAN-Based Analysis of Fiber-Reinforced Geopolymer Concrete

## Overview

This repository contains a machine learning pipeline for predicting the compressive strength of fiber-reinforced geopolymer concrete, including dataset construction, preprocessing, baseline ML modeling, and GAN-based data augmentation.

This work is part of research related to the following paper:

**[1]** M. Zhang, P. Guo, X. Tan, W. Meng, Y. Bao, *Cradle-to-gate assessment and optimization of sustainable geopolymer concrete*.  
*Journal of Cleaner Production*, 2026, 538: 147387.  
https://www.sciencedirect.com/science/article/pii/S0959652625027441

---

## What I Implemented

### 1) Structured Dataset Construction
- Collected 2,300+ experimental data points from published literature
- Organized key parameters into a structured tabular dataset:
  - Composition parameters
  - Fiber parameters
  - Curing parameters
- Performed basic cleaning and preprocessing for modeling (numeric conversion, feature/label extraction)

### 2) Machine Learning Models for Strength Prediction
Implemented and compared multiple regression models for compressive strength prediction, including:
- Linear Regression
- Decision Tree
- Random Forest
- MLP (Neural Network)
- XGBoost
- LightGBM

Evaluation metrics include MAE, MSE, RMSE, MAPE, and R².

### 3) GAN-Based Data Augmentation
- Applied a GAN-based approach to generate synthetic samples
- Conducted augmentation-ratio experiments
- Evaluated the impact of synthetic data on prediction accuracy and robustness

---

## Repository Structure

```text
ml-prediction-geopolymer-concrete/
├── data/
│   ├── origin_data.xlsx
│   └── origin_data.csv
├── notebooks/
│   ├── geopolymer_ml_pipeline.ipynb
│   └── main_pipeline.py
├── src/
│   ├── data_process.py
│   ├── model_train.py
│   ├── data_augmentation.py
│   ├── plotter.py
│   └── save_utils.py
└── results/
    ├── augmentation_experiment.csv
    └── (representative figures)
