"""
model_selection.py
------------------
Loads the trained models and generates comparison plots so you can
see which model won and why. This is what you screenshot for your
portfolio and GitHub README.

HOW TO RUN:
    python src/model_selection.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.model_selection import learning_curve, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import FEATURE_COLS, PLOTS, ensure_dirs, load, section

# ── Load everything ──────────────────────────────────────────────────
section("MODEL SELECTION — Evaluation & Plots")
ensure_dirs()

X_train, X_test, y_train, y_test = load("train_test_split.pkl")
selected_features                 = load("selected_features.pkl")
ensemble                          = load("model1.pkl")
rf                                = load("model2.pkl")

Xtr = X_train[selected_features]
Xte = X_test[selected_features]

# Predictions from both models
ens_proba = ensemble.predict_proba(Xte)[:, 1]
ens_pred  = ensemble.predict(Xte)
rf_proba  = rf.predict_proba(Xte)[:, 1]
rf_pred   = rf.predict(Xte)

ens_auc = roc_auc_score(y_test, ens_proba)
rf_auc  = roc_auc_score(y_test, rf_proba)

# Use whichever is better as "best"
if ens_auc >= rf_auc:
    best_proba, best_pred, best_name = ens_proba, ens_pred, "Ensemble (Majority Vote)"
else:
    best_proba, best_pred, best_name = rf_proba,  rf_pred,  "Random Forest"
best_auc = roc_auc_score(y_test, best_proba)

print(f"  Ensemble AUC : {ens_auc:.4f}")
print(f"  Random Forest AUC : {rf_auc:.4f}")
print(f"  Using: {best_name}")

# ── 6-panel figure ───────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
fig.suptitle(
    "Sephora 'Worth It?' Predictor — Model Selection Report",
    fontsize=15, fontweight="bold", y=1.01
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. ROC curve
ax = fig.add_subplot(gs[0, 0])
for proba, label, color in [
    (ens_proba, f"Ensemble  AUC={ens_auc:.3f}", "#c0392b"),
    (rf_proba,  f"RandomForest  AUC={rf_auc:.3f}", "#2980b9"),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, lw=2, label=label)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.fill_between(*roc_curve(y_test, best_proba)[:2], alpha=0.07, color="#c0392b")
ax.set_title("ROC Curve"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# 2. Precision-Recall curve
ax = fig.add_subplot(gs[0, 1])
prec, rec, _ = precision_recall_curve(y_test, best_proba)
ap = average_precision_score(y_test, best_proba)
ax.plot(rec, prec, color="#8e44ad", lw=2, label=f"AP={ap:.3f}")
ax.fill_between(rec, prec, alpha=0.1, color="#8e44ad")
ax.axhline(y_test.mean(), color="gray", linestyle="--", lw=0.8, label="Baseline")
ax.set_title("Precision-Recall Curve")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(fontsize=8)

# 3. Confusion matrix
ax = fig.add_subplot(gs[0, 2])
cm = confusion_matrix(y_test, best_pred)
ConfusionMatrixDisplay(cm, display_labels=["Overhyped", "Worth It"]).plot(
    ax=ax, colorbar=False, cmap="Purples"
)
ax.set_title(f"Confusion Matrix\n{best_name}")

# 4. Feature importance (RF)
ax = fig.add_subplot(gs[1, 0])
imp = pd.Series(rf.feature_importances_, index=selected_features).sort_values()
colors = ["#e74c3c" if i == imp.idxmax() else "#3498db" for i in imp.index]
ax.barh(imp.index, imp.values, color=colors)
ax.set_title("Feature Importance (Random Forest)")
ax.set_xlabel("Importance score")

# 5. Learning curves
ax = fig.add_subplot(gs[1, 1])
sizes, tr_sc, val_sc = learning_curve(
    rf, Xtr, y_train,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="roc_auc",
    train_sizes=np.linspace(0.1, 1.0, 8),
    n_jobs=-1,
)
ax.plot(sizes, tr_sc.mean(1),  "o-", color="#e74c3c", label="Train AUC")
ax.fill_between(sizes, tr_sc.mean(1)-tr_sc.std(1), tr_sc.mean(1)+tr_sc.std(1), alpha=0.12, color="#e74c3c")
ax.plot(sizes, val_sc.mean(1), "o-", color="#2ecc71", label="Val AUC")
ax.fill_between(sizes, val_sc.mean(1)-val_sc.std(1), val_sc.mean(1)+val_sc.std(1), alpha=0.12, color="#2ecc71")
ax.set_title("Learning Curves\n(bias vs variance)"); ax.set_xlabel("Training size")
ax.set_ylabel("AUC"); ax.legend()

# 6. Score distribution
ax = fig.add_subplot(gs[1, 2])
ax.hist(best_proba[y_test == 0], bins=40, alpha=0.6, color="#e74c3c", label="Overhyped (true)")
ax.hist(best_proba[y_test == 1], bins=40, alpha=0.6, color="#2ecc71", label="Worth It (true)")
ax.axvline(0.5, color="black", linestyle="--", lw=1, label="Threshold 0.5")
ax.set_title("Score Distribution"); ax.set_xlabel("Predicted probability")
ax.legend(fontsize=8)

out_path = os.path.join(PLOTS, "model_selection_report.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n  Saved → outputs/plots/model_selection_report.png")

# ── Console summary ──────────────────────────────────────────────────
report = classification_report(
    y_test, best_pred,
    target_names=["Overhyped", "Worth It"],
    output_dict=True
)
gap = tr_sc.mean(1)[-1] - val_sc.mean(1)[-1]

print(f"""
┌──────────────────────────────────────────────────────┐
│  RESULTS SUMMARY                                     │
├──────────────────────────────────────────────────────┤
│  Best model   : {best_name:<36}│
│  Test AUC     : {best_auc:.4f}                               │
│  Precision    : {report['Worth It']['precision']:.4f}  (of predicted worth-it, % correct)  │
│  Recall       : {report['Worth It']['recall']:.4f}  (of actual worth-it, % found)    │
│  Train-Val gap: {gap:.4f}  {'(well fitted ✓)' if gap < 0.05 else '(some overfitting — consider more regularisation)'}
└──────────────────────────────────────────────────────┘

→ YOUR RESUME METRIC: "Achieved {best_auc:.0%} AUC-ROC on held-out test set"
""")

print("  Run next:  python src/tune_model.py")
