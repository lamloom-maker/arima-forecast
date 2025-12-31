from flask import Flask
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

# ====================================
# Load data
# ====================================
DATA_PATH = "World_MerchantFleet_CLEAN.csv"
df = pd.read_csv(DATA_PATH)

# Filter World + Total fleet
df_world = df[
    (df["Economy Label"] == "World") &
    (df["ShipType Label"] == "Total fleet")
].sort_values("Year")

series = df_world["DWT_million"].astype(float)

# ====================================
# Train ARIMA
# ====================================
model = ARIMA(series, order=(1, 1, 1))
model_fit = model.fit()

# ====================================
# Simple route (test)
# ====================================
@app.route("/")
def home():
    forecast = model_fit.forecast(steps=5).tolist()
    return f"<h2>Forecast: {forecast}</h2>"

if __name__ == "__main__":
    app.run()

