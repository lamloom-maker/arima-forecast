from flask import Flask, render_template, request
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

DATA_PATH = "World_MerchantFleet_CLEAN.csv"

# =========================
# Load data once
# =========================
df = pd.read_csv(DATA_PATH)

# ✅ EXACT columns from your dataset
ECONOMIES = sorted(df["Economy Label"].dropna().unique())
SHIP_TYPES = sorted(df["ShipType Label"].dropna().unique())

@app.route("/", methods=["GET", "POST"])
def index():
    forecast = None
    years = None

    selected_economy = ECONOMIES[0]
    selected_ship = SHIP_TYPES[0]
    steps = 5

    if request.method == "POST":
        selected_economy = request.form.get("economy")
        selected_ship = request.form.get("ship_type")
        steps = int(request.form.get("steps", 5))

        df_filtered = df[
            (df["Economy Label"] == selected_economy) &
            (df["ShipType Label"] == selected_ship)
        ].sort_values("Year")

        # 🔒 Safety check
        if len(df_filtered) >= 5:
            series = df_filtered["DWT_million"].astype(float)

            model = ARIMA(series, order=(1, 1, 1))
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=steps).tolist()

            last_year = int(df_filtered["Year"].max())
            years = list(range(last_year + 1, last_year + 1 + steps))

    return render_template(
        "index.html",
        economies=ECONOMIES,
        ship_types=SHIP_TYPES,
        selected_economy=selected_economy,
        selected_ship=selected_ship,
        steps=steps,
        forecast=forecast,
        years=years
    )

if __name__ == "__main__":
    app.run()
