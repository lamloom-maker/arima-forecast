from flask import Flask, render_template, request
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.metrics import mean_squared_error

# ===============================
# Metrics
# ===============================
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_pe(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

app = Flask(__name__)

# ===============================
# Load data
# ===============================
DATA_PATH = "World_MerchantFleet_CLEAN.csv"
df = pd.read_csv(DATA_PATH)

# Dropdown values (from your real columns)
ECONOMIES = sorted(df["Economy Label"].dropna().unique())
SHIP_TYPES = sorted(df["ShipType Label"].dropna().unique())

# ===============================
# Main route
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    forecast = None
    years = None
    rmse = None
    pe = None

    selected_economy = ECONOMIES[0]
    selected_ship = SHIP_TYPES[0]
    steps = 5

    if request.method == "POST":
        selected_economy = request.form.get("economy")
        selected_ship = request.form.get("ship_type")
        steps = int(request.form.get("steps", 5))

        # Filter data
        df_filtered = df[
            (df["Economy Label"] == selected_economy) &
            (df["ShipType Label"] == selected_ship)
        ].sort_values("Year")

        series = df_filtered["DWT_million"].astype(float)

        # ===============================
        # Train / Test split
        # ===============================
        test_size = min(5, len(series) // 3)
        train = series[:-test_size]
        test = series[-test_size:]

        # Train ARIMA
        model = ARIMA(train, order=(1, 1, 1))
        model_fit = model.fit()

        # Predict test period
        pred_test = model_fit.forecast(steps=len(test))

        # Metrics
        rmse = round(calculate_rmse(test, pred_test), 2)
        pe = round(calculate_pe(test, pred_test), 2)

        # Forecast future
        forecast = model_fit.forecast(steps=steps).tolist()

        last_year = int(df_filtered["Year"].max())
        years = list(range(last_year + 1, last_year + 1 + steps))

    return render_template(
        "index.html",
        forecast=forecast,
        years=years,
        rmse=rmse,
        pe=pe,
        economies=ECONOMIES,
        ship_types=SHIP_TYPES,
        selected_economy=selected_economy,
        selected_ship=selected_ship,
        steps=steps
    )

if __name__ == "__main__":
    app.run(debug=True)
