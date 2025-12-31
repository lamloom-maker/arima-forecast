from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open('arima_model.pkl', 'rb') as f:
    model_fit = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    forecast = None

    if request.method == 'POST':
        steps = int(request.form.get('steps', 1))
        forecast = model_fit.forecast(steps=steps).tolist()

    return render_template('index.html', forecast=forecast)

if __name__ == '__main__':
    app.run(debug=True)
