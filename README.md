# PickPerfect - Product Recommendation System

PickPerfect is a Flask-based product recommendation system combining popularity, Neural Collaborative Filtering (NCF), content similarity, explainable AI, and a session-based cart / demo Buy Now flow.

## Current Features
- Popularity-based recommendations
- Time-aware Popularity Now recommendations
- Neural Collaborative Filtering (NCF)
- Content-based product similarity
- Hybrid recommendation ranking
- AI score with automatic weight renormalization when signals are unavailable
- Lightweight product-specific XAI explanations
- Session cart with quantity update/remove
- Buy Now demo checkout
- Indian Rupee price display
- StockCode/Product ID display

## Main Files
```
app.py
ai_score.py
xai.py
app_home.html
app_recommend.html
app_cart.html
app_checkout.html
app_order_success.html
buy_now.html
order_success.html
style.css
app_style.css
Recommendation.ipynb
cleaned_data.csv
model_hybrid.pkl
requirements.txt
```

## Run Locally
```bash
pip install -r requirements.txt
python app.py
```

For Render, use:
```bash
gunicorn app:app
```

The application uses the repository's existing `cleaned_data.csv` so the deployment does not require the larger raw dataset from the development ZIP.

## Demo Checkout
Cart and Buy Now are demonstration flows only. No real payment is processed.

## Author
Ujjwal Kumar
