from flask import Flask, render_template, request
import os
import pickle
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

MODEL_PATH = "arima_model.pkl"
DATA_PATH = "World_MerchantFleet-supply.csv"

# ====================================
# 1) Load and prepare real data
# ====================================
df = pd.read_csv(DATA_PATH)

# 🔴 عدّلي القيم حسب بياناتك
df = df[df["Economy"] == "World"]
df = df[df["ShipType"] == "Total Fleet"]

df = df.sort_values("Year")

series = df["DWT_million"].values

# ====================================
# 2) Load or train ARIMA model
# ====================================
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model_fit = pickle.load(f)
else:
    model = ARIMA(series, order=(1, 1, 1))
    model_fit = model.fit()
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_fit, f)

# ====================================
# 3) Routes
# ====================================
@app.route("/", methods=["GET", "POST"])
def index():
    forecast = None
    if request.method == "POST":
        steps = int(request.form.get("steps", 1))
        forecast = model_fit.forecast(steps=steps).tolist()
    return render_template("index.html", forecast=forecast)

if __name__ == "__main__":
    app.run()

