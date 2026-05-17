"""
utils.py
--------
Shared helper functions used across all scripts.
Keeps the other files clean — no repeated code.
"""

import os
import joblib
import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT   = os.path.join(ROOT, "input")
MODELS  = os.path.join(ROOT, "models")
OUTPUTS = os.path.join(ROOT, "outputs")
PLOTS   = os.path.join(OUTPUTS, "plots")

# Column names the model trains on (in order)
FEATURE_COLS = [
    "price_usd",
    "limited_edition",
    "new",
    "online_only",
    "out_of_stock",
    "sephora_exclusive",
    "reviews",
    "category_encoded",
    "brand_avg_rating",
]


def ensure_dirs():
    """Create output folders if they don't exist yet."""
    for path in [MODELS, OUTPUTS, PLOTS]:
        os.makedirs(path, exist_ok=True)


def save(obj, filename):
    """Save any Python object to models/ folder using joblib."""
    ensure_dirs()
    path = os.path.join(MODELS, filename)
    joblib.dump(obj, path)
    print(f"  Saved → models/{filename}")


def load(filename):
    """Load a saved object from models/ folder."""
    path = os.path.join(MODELS, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nCannot find models/{filename}\n"
            "Make sure you have run the previous scripts in order:\n"
            "  python src/train.py  →  python src/model_selection.py  →  etc."
        )
    return joblib.load(path)


def save_output(df_or_array, filename):
    """Save a DataFrame or array to outputs/ folder as CSV."""
    ensure_dirs()
    path = os.path.join(OUTPUTS, filename)
    if isinstance(df_or_array, pd.DataFrame):
        df_or_array.to_csv(path, index=False)
    else:
        pd.DataFrame(df_or_array).to_csv(path, index=False)
    print(f"  Saved → outputs/{filename}")


def section(title):
    """Print a nice section header to the terminal."""
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)
