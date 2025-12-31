from flask import Flask, render_template, request
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

DATA_PATH = "World_MerchantFleet_CLEAN.csv"

# ====================================
# Load data once
# ====================================
df = pd.read_csv(DATA_PATH)

# Ship types available
SHIP_TYPES = sorted(df["ShipType Label"].unique())

@app.route("/", methods=["GET", "POST"])
def index():
    forecast = None
    selected_ship = "Total fleet"
    steps = 5

    if request.method == "POST":
        selected_ship = request.form.get("ship_type")
        steps = int(request.form.get("steps", 5))

        df_filtered = df[
            (df["Economy Label"] == "World") &
            (df["ShipType Label"] == selected_ship)
        ].sort_values("Year")

        series = df_filtered["DWT_million"].astype(float)

        model = ARIMA(series, order=(1, 1, 1))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=steps).tolist()

    return render_template(
        "index.html",
        forecast=forecast,
        ship_types=SHIP_TYPES,
        selected_ship=selected_ship,
        steps=steps
    )

if __name__ == "__main__":
    app.run()


