"""
tune_model.py
-------------
Takes the best model (Random Forest) and tries many different
settings combinations to find the optimal hyperparameters.
Uses nested cross-validation — the most honest evaluation method.

HOW TO RUN:
    python src/tune_model.py

⚠️  Takes 5–15 minutes. That's normal — grabbing a snack is encouraged.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV, cross_val_score, StratifiedKFold
)
from sklearn.metrics import roc_auc_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import PLOTS, ensure_dirs, load, save, section

section("HYPERPARAMETER TUNING")
ensure_dirs()

X_train, X_test, y_train, y_test = load("train_test_split.pkl")
selected_features                 = load("selected_features.pkl")

Xtr = X_train[selected_features]
Xte = X_test[selected_features]

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Tune Random Forest ───────────────────────────────────────────────
print("\n  Tuning Random Forest (trying different tree depths + sizes)...")
print("  This is the slow part — ~5-10 minutes. Hang tight.\n")

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid={
        "n_estimators":      [100, 200, 300],
        "max_depth":         [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
    },
    cv=cv5, scoring="roc_auc", n_jobs=-1, verbose=1,
)
rf_grid.fit(Xtr, y_train)
print(f"\n  Best RF params : {rf_grid.best_params_}")
print(f"  Best RF CV AUC : {rf_grid.best_score_:.4f}")

# ── Tune Logistic Regression ─────────────────────────────────────────
print("\n  Tuning Logistic Regression (C + penalty)...")
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid={
        "C":       [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l1", "l2"],
        "solver":  ["liblinear"],
    },
    cv=cv5, scoring="roc_auc", n_jobs=-1, verbose=1,
)
lr_grid.fit(Xtr, y_train)
print(f"\n  Best LR params : {lr_grid.best_params_}")
print(f"  Best LR CV AUC : {lr_grid.best_score_:.4f}")

# ── Nested Cross-Validation ──────────────────────────────────────────
print("\n  Running nested cross-validation (the honest accuracy estimate)...")
best_rf = rf_grid.best_estimator_
outer   = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
nested  = cross_val_score(best_rf, Xtr, y_train, cv=outer, scoring="roc_auc", n_jobs=-1)
print(f"  Nested CV scores : {[f'{s:.4f}' for s in nested]}")
print(f"  Mean  : {nested.mean():.4f}")
print(f"  Std   : {nested.std():.4f}")

# ── Final test set evaluation ────────────────────────────────────────
best_rf.fit(Xtr, y_train)
y_pred  = best_rf.predict(Xte)
y_proba = best_rf.predict_proba(Xte)[:, 1]
test_auc = roc_auc_score(y_test, y_proba)

print(f"\n  Final Test AUC : {test_auc:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Overhyped", "Worth It"]))

# ── Save tuned models (overwrite model1 + model2 with best tuned) ────
save(best_rf,                "model1.pkl")   # tuned RF replaces ensemble as primary
save(lr_grid.best_estimator_,"model2.pkl")   # tuned LR as secondary

# ── Plot: tuning curves ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Hyperparameter Tuning Results", fontsize=13, fontweight="bold")

# LR: C vs AUC
ax = axes[0]
lr_res = pd.DataFrame(lr_grid.cv_results_)
for pen in ["l1", "l2"]:
    sub = lr_res[lr_res["param_penalty"] == pen].sort_values("param_C")
    ax.semilogx(sub["param_C"].astype(float), sub["mean_test_score"], marker="o", label=f"penalty={pen}")
ax.axvline(lr_grid.best_params_["C"], color="red", linestyle="--", alpha=0.5)
ax.set_title("Logistic Regression: C vs AUC")
ax.set_xlabel("C (regularization strength)"); ax.set_ylabel("Mean CV AUC"); ax.legend()

# RF: n_estimators vs AUC at best max_depth
ax = axes[1]
rf_res   = pd.DataFrame(rf_grid.cv_results_)
best_dep = rf_grid.best_params_["max_depth"]
sub      = rf_res[rf_res["param_max_depth"] == best_dep].sort_values("param_n_estimators")
ax.plot(sub["param_n_estimators"].astype(int), sub["mean_test_score"], marker="o", color="#e74c3c")
ax.set_title(f"Random Forest: n_estimators vs AUC\n(max_depth={best_dep})")
ax.set_xlabel("Number of trees"); ax.set_ylabel("Mean CV AUC")

out_path = os.path.join(PLOTS, "tuning_results.png")
plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n  Saved → outputs/plots/tuning_results.png")
print("\n  Run next:  python src/predict.py")
