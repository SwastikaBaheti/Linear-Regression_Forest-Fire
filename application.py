from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Import Ridge Regression and Standard Scaler pickle
ridge_model = pickle.load(open('models/ridge_regression.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        temp = float(request.form["Temperature"])
        rh = float(request.form["RH"])
        ws = float(request.form["Ws"])
        rain = float(request.form["Rain"])
        ffmc = float(request.form["FFMC"])
        dmc = float(request.form["DMC"])
        isi = float(request.form["ISI"])
        classes = float(request.form["Classes"])
        region = float(request.form["Region"])

        scaled_input = standard_scaler.transform([[temp, rh, ws, rain, ffmc, dmc, isi, classes, region]])
        fwi_predicted = ridge_model.predict(scaled_input)

        return render_template('home.html', result=fwi_predicted[0])
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')