# optimized_stacked_pathloss_high_accuracy.py
"""
High Accuracy Version (Option A)
 - Designed to beat benchmark RMSE/MAE/MAPE/R2
 - More Optuna trials
 - Stronger MLP meta-learner (3-layer NN)
 - Better Stacking CV
 - Full CPU parallelization
 - LightGBM removed (Python 3.13 safe)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from functools import partial

import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb

# ---------------------------
# CONFIG (High Accuracy)
# ---------------------------
RNG = 42
random.seed(RNG)
np.random.seed(RNG)

DATA_PATH = "D:/5th sem/ML/5g-South Asia.csv"
OUT_DIR = "D:/5th sem/ML/optimized_high_accuracy_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

N_TRIALS = 40        # Higher tuning for base models
META_TRIALS = 25     # Stronger meta-learner tuning
CV_FOLDS = 5         # 5-fold CV (accurate + reasonable)
STACK_CV = 5         # Stronger stacking CV (big improvement)
N_JOBS = -1          # Full CPU usage

# ---------------------------
# Metrics
# ---------------------------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ---------------------------
# Load + Feature engineering
# ---------------------------
df = pd.read_csv(DATA_PATH)
drop_cols = ["Seasonal Variation (Data Source)", "Simulation Run Number"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

target = "Path Loss (dB)"

def add_features(df):
    df = df.copy()
    if "Frequency (MHz)" in df.columns:
        df["Freq_GHz"] = df["Frequency (MHz)"] / 1000
    if "Distance (m)" in df.columns:
        df["Log_Distance"] = np.log1p(df["Distance (m)"])
    if "Tx Height (m)" in df.columns and "Rx Height (m)" in df.columns:
        df["Height_Diff"] = df["Tx Height (m)"] - df["Rx Height (m)"]
        df["Avg_Height"] = 0.5 * (df["Tx Height (m)"] + df["Rx Height (m)"])
    if "Distance (m)" in df.columns and "Frequency (MHz)" in df.columns:
        df["FSPL"] = (
            20*np.log10(df["Distance (m)"].replace(0, 1e-6)) +
            20*np.log10(df["Frequency (MHz)"].replace(0, 1e-6)) + 32.44
        )
    if "Log_Distance" in df.columns and "Freq_GHz" in df.columns:
        df["LogDist_x_Freq"] = df["Log_Distance"] * df["Freq_GHz"]
    return df

X = add_features(df.drop(columns=[target]))
y = df[target]

X = X.apply(pd.to_numeric, errors="coerce")
mask = X.isna().any(axis=1)
X = X[~mask]; y = y[~mask]

# ---------------------------
# Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RNG
)

num_cols = X_train.columns.tolist()
preprocessor = ColumnTransformer([("num", StandardScaler(), num_cols)], remainder="passthrough")

# ---------------------------
# RMSE CV Wrapper
# ---------------------------
def cv_rmse_estimator(estimator, X, y, cv=CV_FOLDS):
    scores = cross_val_score(estimator, X, y,
                             scoring="neg_root_mean_squared_error",
                             cv=cv, n_jobs=1)
    return -scores.mean()

# ---------------------------
# Optuna Objectives (High Accuracy)
# ---------------------------
def objective_rf(trial, X, y):
    params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [300, 500, 800]),
        "max_depth": trial.suggest_categorical("max_depth", [12, 20, None]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"])
    }
    model = RandomForestRegressor(random_state=RNG, n_jobs=1, **params)
    pipe = Pipeline([("pre", preprocessor), ("model", model)])
    return cv_rmse_estimator(pipe, X, y)

def objective_svr(trial, X, y):
    C = trial.suggest_loguniform("C", 1e0, 1e3)
    gamma = trial.suggest_categorical("gamma", ["scale", 0.01, 0.001])
    eps = trial.suggest_float("epsilon", 0.001, 0.2)
    model = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=eps)
    pipe = Pipeline([("pre", preprocessor), ("model", model)])
    return cv_rmse_estimator(pipe, X, y)

def objective_xgb(trial, X, y):
    params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [200, 400, 600]),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 1e-1),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 1e-1),
        "n_jobs": 1,
        "random_state": RNG
    }
    model = xgb.XGBRegressor(**params)
    pipe = Pipeline([("pre", preprocessor), ("model", model)])
    return cv_rmse_estimator(pipe, X, y)

def objective_gb(trial, X, y):
    params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [200, 400, 600]),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.15),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0)
    }
    model = GradientBoostingRegressor(random_state=RNG, **params)
    pipe = Pipeline([("pre", preprocessor), ("model", model)])
    return cv_rmse_estimator(pipe, X, y)

# ---------------------------
# Optuna tuning
# ---------------------------
print("Tuning RandomForest...")
study_rf = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RNG))
study_rf.optimize(partial(objective_rf, X=X_train, y=y_train), n_trials=N_TRIALS)

print("Tuning SVR...")
study_svr = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RNG))
study_svr.optimize(partial(objective_svr, X=X_train, y=y_train), n_trials=N_TRIALS)

print("Tuning XGBoost...")
study_xgb = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RNG))
study_xgb.optimize(partial(objective_xgb, X=X_train, y=y_train), n_trials=N_TRIALS)

print("Tuning GradientBoosting...")
study_gb = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RNG))
study_gb.optimize(partial(objective_gb, X=X_train, y=y_train), n_trials=N_TRIALS)

# ---------------------------
# Build optimized models
# ---------------------------
best_rf = RandomForestRegressor(**study_rf.best_trial.params, random_state=RNG, n_jobs=N_JOBS)
best_svr = SVR(**study_svr.best_trial.params)
best_xgb = xgb.XGBRegressor(**study_xgb.best_trial.params, random_state=RNG, n_jobs=N_JOBS)
best_gb = GradientBoostingRegressor(**study_gb.best_trial.params, random_state=RNG)

estimators = [
    ("rf", best_rf),
    ("svr", best_svr),
    ("xgb", best_xgb),
    ("gb", best_gb)
]

# ---------------------------
# Meta-learner tuning (High Accuracy)
# ---------------------------
def objective_meta(trial, X, y):
    h1 = trial.suggest_categorical("h1", [64, 96, 128])
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    alpha = trial.suggest_loguniform("alpha", 1e-6, 1e-2)

    meta = MLPRegressor(
        hidden_layer_sizes=(h1, h1, h1),  # 3 layers → much better accuracy
        learning_rate_init=lr,
        alpha=alpha,
        activation="relu",
        max_iter=1000,
        early_stopping=True,
        random_state=RNG
    )

    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=meta,
        cv=STACK_CV,
        passthrough=True,
        n_jobs=N_JOBS
    )

    pipe = Pipeline([("pre", preprocessor), ("stack", stack)])
    return cv_rmse_estimator(pipe, X, y)

print("Tuning Meta-Learner...")
study_meta = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RNG))
study_meta.optimize(partial(objective_meta, X=X_train, y=y_train), n_trials=META_TRIALS)

meta_params = study_meta.best_trial.params
meta = MLPRegressor(
    hidden_layer_sizes=(meta_params["h1"], meta_params["h1"], meta_params["h1"]),
    learning_rate_init=meta_params["lr"],
    alpha=meta_params["alpha"],
    max_iter=1200,
    early_stopping=True,
    random_state=RNG
)

# ---------------------------
# Final Stacking Model
# ---------------------------
stack = StackingRegressor(
    estimators=estimators,
    final_estimator=meta,
    cv=STACK_CV,
    passthrough=True,
    n_jobs=N_JOBS
)

pipeline = Pipeline([("pre", preprocessor), ("stack", stack)])

print("Training final high accuracy model...")
pipeline.fit(X_train, y_train)

# ---------------------------
# Test evaluation
# ---------------------------
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape_v = mape(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== HIGH ACCURACY TEST RESULTS =====")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MAPE : {mape_v:.4f}%")
print(f"R2   : {r2:.4f}")

# ---------------------------
# Save Artifacts
# ---------------------------
joblib.dump(pipeline, os.path.join(OUT_DIR, "high_accuracy_pipeline.pkl"))
pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
    os.path.join(OUT_DIR, "predictions_high_accuracy.csv"), index=False)

# ---------------------------
# Plots
# ---------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6, s=12)
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([mn, mx], [mn, mx], "--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("High Accuracy Parity Plot")
plt.savefig(os.path.join(OUT_DIR, "parity_high_accuracy.png"), dpi=200)
plt.close()

plt.figure(figsize=(7,4))
plt.hist(y_test - y_pred, bins=40)
plt.title("High Accuracy Residuals")
plt.savefig(os.path.join(OUT_DIR, "residual_high_accuracy.png"), dpi=200)
plt.close()

print("\nArtifacts saved to:", OUT_DIR)
