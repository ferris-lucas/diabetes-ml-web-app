import flask
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import joblib
import pickle

#Declare the app with constructor
app = Flask(__name__)

#Load model from serialized file model
loaded_model = joblib.load('model.pkl')

#Decorator to add the function root
@app.route("/")
def root():
# Calls the render_template and points to html
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def make_prediction():

    if request.method == 'POST':
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        blood_pressure = request.form['blood_pressure']
        skin_thickness = request.form['skin_thickness']
        insulin = request.form['insulin']
        BMI = request.form['BMI']
        DPF = request.form['DPF']
        age = request.form['age']

        user_data = {'pregnancies': pregnancies,
                     'glucose': glucose,
                     'blood_pressure': blood_pressure,
                     'skin_thickness': skin_thickness,
                     'insulin': insulin,
                     'BMI': BMI,
                     'DPF': DPF,
                     'age': age
                     }

        user_input = pd.DataFrame(user_data, index=[0])

        [prediction] = loaded_model.predict(user_input)

    if prediction = 1:
        msg = "A previsão é de diagnóstico positivo, tem diabetes."
    else:
        msg = "A previsão é de diagnóstico negativo, não tem diabetes."

    return render_template("index.html", prediction_text = msg)

if __name__ == '__main__':
    app.run(debug=True)
