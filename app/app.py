"""
app/app.py
----------
Flask web app. Imports predict.py so there's zero duplicated code.

HOW TO RUN (from the project root folder):
    python app/app.py

Then open: http://127.0.0.1:5000
"""

import sys
import os

# Add src/ to path so we can import predict.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from flask import Flask, request, render_template, jsonify
from predict import load_pipeline, predict_product

app = Flask(__name__)

# Load model once when the app starts (not on every request)
print("Loading model...")
pipeline = load_pipeline()
print("Ready! Open http://127.0.0.1:5000")

# Get known categories for the dropdown
le_cat     = pipeline[3]
CATEGORIES = list(le_cat.classes_)


@app.route("/")
def home():
    return render_template("index.html", categories=CATEGORIES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        product = {
            "price_usd":         request.form.get("price", 30),
            "category":          request.form.get("category", "Moisturizers"),
            "reviews":           request.form.get("reviews", 100),
            "brand_avg_rating":  request.form.get("brand_avg_rating", 4.0),
            "limited_edition":   1 if request.form.get("limited_edition") else 0,
            "new":               1 if request.form.get("new") else 0,
            "online_only":       1 if request.form.get("online_only") else 0,
            "out_of_stock":      1 if request.form.get("out_of_stock") else 0,
            "sephora_exclusive": 1 if request.form.get("sephora_exclusive") else 0,
        }
        result = predict_product(product, *pipeline)
        
        return render_template("result.html", **result)
    except Exception as e:
        return render_template("result.html", error=str(e))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API — test with curl or Postman."""
    try:
        result = predict_product(request.get_json(), *pipeline)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("\n" + "=" * 45)
    print("  Flask app running!")
    print("  Visit: http://127.0.0.1:5000")
    print("  Press CTRL+C to stop")
    print("=" * 45 + "\n")
    app.run(debug=True, port=5000)
