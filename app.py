import pandas as pd
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

loaded_model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html',
                           prediction_text='Altere os valores papra gerar previsão',
                           v1=3,
                           v2=90,
                           v3=90,
                           v4=50,
                           v5=95,
                           v6=30,
                           v7=0.500,
                           v8=50)

@app.route('/predict', methods=['POST',])
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

    if prediction == 1:
        msg = "A previsão é de que o paciente tem diabetes."
    else:
        msg = "A previsão é de que o paciente não tem diabetes."

    return render_template("index.html",   prediction_text=msg,
                                           v1=pregnancies,
                                           v2=glucose,
                                           v3=blood_pressure,
                                           v4=skin_thickness,
                                           v5=insulin,
                                           v6=BMI,
                                           v7=DPF,
                                           v8=age)
if __name__ == '__main__':
    app.run(debug=True)