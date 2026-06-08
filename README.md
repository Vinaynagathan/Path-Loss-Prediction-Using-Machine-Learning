# Path Loss Prediction Using Machine Learning

## Overview
This project presents a machine learning-based framework for predicting path loss in 5G mmWave wireless communication systems.

## Dataset
- NYUSIM 3.0 mmWave Wireless Propagation Dataset
- 2,835 samples
- Environmental and propagation parameters

## Models Used
- Random Forest Regressor
- Support Vector Regression (SVR)
- Gradient Boosting Regressor
- Stacking Ensemble with Ridge Regression Meta-Learner

## Results

| Metric | Value |
|----------|----------|
| R² Score | 0.919 |
| RMSE | 4.21 dB |
| MAE | 2.96 dB |
| MAPE | 1.85% |

## Technologies
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

## Key Contributions
- Developed a stacking ensemble framework for path loss prediction.
- Combined Random Forest, SVR, and Gradient Boosting models.
- Achieved improved accuracy compared to individual ML models.
- Applied to 5G mmWave wireless communication environments.

## Author
N Vinay
