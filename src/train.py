"""
train.py
--------
Loads the raw Kaggle data, preprocesses it, selects features,
trains all 4 classifiers + the ensemble, and saves everything
to the models/ folder.

HOW TO RUN (from the project root folder):
    python src/train.py
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score

# Add src/ to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    FEATURE_COLS, INPUT, MODELS, PLOTS,
    ensure_dirs, save, save_output, section,
)


# ════════════════════════════════════════════════════════
# 1. LOAD DATA
# ════════════════════════════════════════════════════════

section("STEP 1 / 4 — Loading data")

products_path = os.path.join(INPUT, "product_info.csv")
reviews_path  = os.path.join(INPUT, "reviews_0-250.csv")

if not os.path.exists(products_path):
    print("\n ERROR: Cannot find input/product_info.csv")
    print(" → Download the dataset from Kaggle and put the CSV files in the input/ folder.")
    print(" → See the guide for step-by-step instructions.")
    sys.exit(1)

products = pd.read_csv(products_path)
reviews  = pd.read_csv(reviews_path)
print(f"  Loaded {len(products):,} products")
print(f"  Loaded {len(reviews):,} reviews")


# ════════════════════════════════════════════════════════
# 2. CREATE TARGET LABEL
# ════════════════════════════════════════════════════════

section("STEP 2 / 4 — Creating target label")

# worth_it = 1  if rating >= 4.0  AND  loves_count >= 500
# worth_it = 0  otherwise (overhyped)
products["worth_it"] = (
    (products["rating"] >= 4.0) &
    (products["loves_count"] >= 500)
).astype(int)

n_pos = products["worth_it"].sum()
n_tot = len(products)
print(f"  Worth It  : {n_pos:,}  ({100*n_pos/n_tot:.1f}%)")
print(f"  Overhyped : {n_tot-n_pos:,}  ({100*(n_tot-n_pos)/n_tot:.1f}%)")


# ════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ════════════════════════════════════════════════════════

section("STEP 3 / 4 — Engineering features")

df = products.copy()

# Fill missing prices with the category median
df["price_usd"] = df.groupby("primary_category")["price_usd"].transform(
    lambda x: x.fillna(x.median())
)
df["price_usd"] = df["price_usd"].fillna(df["price_usd"].median())

# Fill missing review counts
df["reviews"] = df["reviews"].fillna(0)

# Encode product category as a number
le_cat = LabelEncoder()
df["category_encoded"] = le_cat.fit_transform(
    df["primary_category"].fillna("Unknown")
)

# Brand average rating (brand reputation signal)
brand_avg = df.groupby("brand_name")["rating"].mean().rename("brand_avg_rating")
df = df.merge(brand_avg, on="brand_name", how="left")
df["brand_avg_rating"] = df["brand_avg_rating"].fillna(df["rating"].mean())

# Boolean flags → 0 / 1
for col in ["limited_edition", "new", "online_only", "out_of_stock", "sephora_exclusive"]:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)
    else:
        df[col] = 0

print("  Features created")

# Build X, y
X = df[FEATURE_COLS].copy()
y = df["worth_it"].copy()

# Impute any remaining NaNs
imputer = SimpleImputer(strategy="median")
X_arr   = imputer.fit_transform(X)
X       = pd.DataFrame(X_arr, columns=FEATURE_COLS)

# Scale features (critical for SVM + Logistic Regression)
scaler  = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS)

print(f"  Final shape: {X_scaled.shape[0]:,} rows × {X_scaled.shape[1]} features")


# ════════════════════════════════════════════════════════
# 4. FEATURE SELECTION (L1 + Random Forest)
# ════════════════════════════════════════════════════════

section("STEP 4 / 4 — Feature selection + training models")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")

# L1 regularization — drops useless features to exactly zero
print("\n  Running L1 feature selection...")
lr_l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=1000, random_state=42)
lr_l1.fit(X_train, y_train)
coef       = pd.Series(lr_l1.coef_[0], index=FEATURE_COLS)
selected_l1 = coef[coef != 0].index.tolist()

# Random Forest importance — drops features below 1% importance
print("  Running Random Forest feature importance...")
rf_sel = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_sel.fit(X_train, y_train)
importances = pd.Series(rf_sel.feature_importances_, index=FEATURE_COLS)
selected_rf = importances[importances > 0.01].index.tolist()

# Union of both methods
selected_features = list(set(selected_l1) | set(selected_rf))
print(f"\n  Selected {len(selected_features)}/{len(FEATURE_COLS)} features: {selected_features}")

# Narrow to selected features
Xtr = X_train[selected_features]
Xte = X_test[selected_features]

# ── Train 4 classifiers ──────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

classifiers = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "SVM":                 SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    "AdaBoost":            AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
}

results = {}
print("\n  Training classifiers:")
for name, clf in classifiers.items():
    t0 = time.time()
    cv_auc = cross_val_score(clf, Xtr, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
    clf.fit(Xtr, y_train)
    test_auc = roc_auc_score(y_test, clf.predict_proba(Xte)[:, 1])
    print(f"    {name:<25}  CV AUC={cv_auc:.4f}   Test AUC={test_auc:.4f}   ({time.time()-t0:.1f}s)")
    results[name] = {"clf": clf, "cv_auc": cv_auc, "test_auc": test_auc}

# ── Ensemble ─────────────────────────────────────────────────────────
print("\n  Building majority-vote ensemble...")
ensemble = VotingClassifier(
    estimators=[(n, results[n]["clf"]) for n in classifiers],
    voting="soft",
)
ens_cv  = cross_val_score(ensemble, Xtr, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
ensemble.fit(Xtr, y_train)
ens_auc = roc_auc_score(y_test, ensemble.predict_proba(Xte)[:, 1])
print(f"    {'Ensemble':<25}  CV AUC={ens_cv:.4f}   Test AUC={ens_auc:.4f}")
results["Ensemble"] = {"clf": ensemble, "cv_auc": ens_cv, "test_auc": ens_auc}

best_name = max(results, key=lambda k: results[k]["test_auc"])
print(f"\n  Best model: {best_name}  (AUC = {results[best_name]['test_auc']:.4f})")


# ── Save everything ──────────────────────────────────────────────────
ensure_dirs()
save(imputer,          "imputer.pkl")
save(scaler,           "scaler.pkl")
save(le_cat,           "label_encoder.pkl")
save(selected_features,"selected_features.pkl")
save((X_train, X_test, y_train, y_test), "train_test_split.pkl")
save(ensemble,         "model1.pkl")           # ensemble = primary model
save(results["Random Forest"]["clf"], "model2.pkl")  # RF alone = secondary

# Save a comparison CSV
comp = pd.DataFrame([
    {"model": n, "cv_auc": r["cv_auc"], "test_auc": r["test_auc"]}
    for n, r in results.items()
])
save_output(comp, "model_comparison.csv")

print("\n  All models saved to models/")
print("  Run next:  python src/model_selection.py")
