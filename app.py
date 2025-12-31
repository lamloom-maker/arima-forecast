from flask import Flask, render_template, request
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

# ===============================
# Load data
# ===============================
DATA_PATH = "World_MerchantFleet_CLEAN.csv"
df = pd.read_csv(DATA_PATH)

# ===============================
# Prepare dropdown values
# ===============================
ECONOMIES = sorted(df["Economy Label"].dropna().unique())
SHIP_TYPES = sorted(df["ShipType Label"].dropna().unique())

# ===============================
# Main route
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    forecast = None
    years = None

    # default selections
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

        # Train ARIMA
        model = ARIMA(series, order=(1, 1, 1))
        model_fit = model.fit()

        # Forecast
        forecast = model_fit.forecast(steps=steps).tolist()

        # Generate future years
        last_year = int(df_filtered["Year"].max())
        years = list(range(last_year + 1, last_year + 1 + steps))

    return render_template(
        "index.html",
        forecast=forecast,
        years=years,
        economies=ECONOMIES,
        ship_types=SHIP_TYPES,
        selected_economy=selected_economy,
        selected_ship=selected_ship,
        steps=steps
    )


if __name__ == "__main__":
    app.run(debug=True)
