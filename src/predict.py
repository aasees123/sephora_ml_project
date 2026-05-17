"""
predict.py
----------
Loads the trained model and makes predictions on new product data.
Two ways to use this:

  1. Command line (quick test):
       python src/predict.py

  2. Import into the Flask app (app/app.py does this automatically)

HOW TO RUN:
    python src/predict.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import FEATURE_COLS, load, section


def load_pipeline():
    """Load model + all preprocessing objects. Called once at startup."""
    model             = load("model1.pkl")
    imputer           = load("imputer.pkl")
    scaler            = load("scaler.pkl")
    le_cat            = load("label_encoder.pkl")
    selected_features = load("selected_features.pkl")
    return model, imputer, scaler, le_cat, selected_features


def predict_product(product_dict, model, imputer, scaler, le_cat, selected_features):
    """
    Given a dictionary of product attributes, return a prediction.

    product_dict keys:
        price_usd         (float)  e.g. 48.0
        category          (str)    e.g. "Moisturizers"
        reviews           (int)    e.g. 2500
        brand_avg_rating  (float)  e.g. 4.2
        limited_edition   (0/1)
        new               (0/1)
        online_only       (0/1)
        out_of_stock      (0/1)
        sephora_exclusive (0/1)

    Returns dict with keys: prediction, probability, confidence, label, message
    """

    # Encode category
    known = list(le_cat.classes_)
    cat   = product_dict.get("category", "Moisturizers")
    cat_encoded = le_cat.transform([cat])[0] if cat in known else 0

    # Build feature row in the correct column order
    row = {
        "price_usd":         float(product_dict.get("price_usd", 30)),
        "limited_edition":   int(product_dict.get("limited_edition", 0)),
        "new":               int(product_dict.get("new", 0)),
        "online_only":       int(product_dict.get("online_only", 0)),
        "out_of_stock":      int(product_dict.get("out_of_stock", 0)),
        "sephora_exclusive": int(product_dict.get("sephora_exclusive", 0)),
        "reviews":           float(product_dict.get("reviews", 100)),
        "category_encoded":  cat_encoded,
        "brand_avg_rating":  float(product_dict.get("brand_avg_rating", 4.0)),
    }

    X = np.array([[row[col] for col in FEATURE_COLS]])

    # Apply same preprocessing as training
    X = imputer.transform(X)
    X = scaler.transform(X)

    # Keep only selected features
    sel_idx = [FEATURE_COLS.index(f) for f in selected_features if f in FEATURE_COLS]
    X = X[:, sel_idx]

    # Predict
    prob = float(model.predict_proba(X)[0][1])
    pred = int(prob >= 0.5)

    # Human-readable output
    if prob >= 0.75:
        message = "Strong buy — the data strongly supports this product."
    elif prob >= 0.5:
        message = "Probably worth trying, but do your own research too."
    elif prob >= 0.25:
        message = "Probably overhyped — the numbers are not convincing."
    else:
        message = "Skip it. The model sees real red flags here."

    return {
        "prediction":  pred,
        "probability": round(prob, 4),
        "confidence":  round(max(prob, 1 - prob) * 100, 1),
        "label":       "Worth It ✓" if pred == 1 else "Overhyped ✗",
        "message":     message,
    }


# ── Quick demo when run directly ────────────────────────────────────
if __name__ == "__main__":
    section("PREDICT — demo mode")

    pipeline = load_pipeline()

    test_products = [
        {
            "name": "La Mer Moisturizing Cream",
            "price_usd": 195, "category": "Moisturizers",
            "reviews": 8200, "brand_avg_rating": 4.5,
            "sephora_exclusive": 0, "limited_edition": 0,
            "new": 0, "online_only": 0, "out_of_stock": 0,
        },
        {
            "name": "Random Hyped New Serum",
            "price_usd": 89, "category": "Serums",
            "reviews": 12, "brand_avg_rating": 3.1,
            "sephora_exclusive": 0, "limited_edition": 1,
            "new": 1, "online_only": 1, "out_of_stock": 1,
        },
        {
            "name": "Charlotte Tilbury Magic Cream",
            "price_usd": 100, "category": "Moisturizers",
            "reviews": 5500, "brand_avg_rating": 4.4,
            "sephora_exclusive": 1, "limited_edition": 0,
            "new": 0, "online_only": 0, "out_of_stock": 0,
        },
    ]

    print(f"\n{'Product':<35} {'Verdict':<15} {'Prob':>6}  {'Confidence':>10}")
    print("-" * 72)

    for p in test_products:
        result = predict_product(p, *pipeline)
        print(
            f"  {p['name']:<33} {result['label']:<15} "
            f"{result['probability']:>6.3f}  {result['confidence']:>8.1f}%"
        )
        print(f"  {'':33} → {result['message']}")
        print()

    print("  Run next:  python app/app.py")
