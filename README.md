SEPHORA "WORTH IT?" — ML PRODUCT PREDICTOR

a machine learning classifier trained on 1M+ sephora reviews to predict whether a beauty product is genuinely loved — or just overhyped. the goal was to go through the full ml pipeline end to end: data cleaning, feature engineering, model comparison, hyperparameter tuning, and a simple web app to serve predictions.

DATA

the model is trained on:

- 8,000+ sephora products with metadata (brand, category, price, rating, etc.)
- 1M+ user reviews (rating, review text, helpfulness votes, etc.)
- both files merged during preprocessing on product id

dataset: sephora products and skincare reviews — https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews (free on kaggle)

FEATURES

the model uses a mix of product-level and review-aggregated features:

- product price and category
- average rating and total review count
- helpfulness vote ratio
- skin type and concern tags (encoded)
- brand-level reputation signal (aggregated from reviews)

ML TECHNIQUES

preprocessing      — SimpleImputer, LabelEncoder, StandardScaler
feature selection  — L1 regularization + Random Forest importance
models             — Logistic Regression, SVM, Random Forest, AdaBoost
ensemble           — soft majority vote (VotingClassifier)
evaluation         — ROC-AUC, Precision-Recall, Confusion Matrix
tuning             — GridSearchCV + Nested Cross-Validation
deployment         — Flask REST API

CURRENT PERFORMANCE

- roc-auc: competitive across all four base models
- ensemble (soft vote) outperforms any single model
- evaluation plots auto-saved to outputs/plots/

HOW TO RUN

setup:
python3 -m venv venv
source venv/bin/activate      # mac/linux
venv\Scripts\activate         # windows
pip install -r requirements.txt

then download the two CSVs from kaggle and drop them in input/.

train:
python src/train.py

evaluate:
python src/model_selection.py

tune (takes ~10 min):
python src/tune_model.py

predict:
python src/predict.py

web app:
python app/app.py

LIMITATIONS

- review text isn't used (just metadata and aggregated signals)
- "worth it" label is derived from ratings, so it inherits that bias
- no temporal features — older vs. newer reviews treated equally
- only covers products listed on sephora

NEXT STEPS

- incorporate nlp on review text (sentiment, keyword features)
- add recency weighting to reviews
- improve the flask UI
- experiment with gradient boosting (xgboost/lightgbm)
```