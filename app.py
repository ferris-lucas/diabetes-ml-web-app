import joblib
import pandas as pd
from flask import Flask, render_template, request

# creates a global Falsk instance sets the variable name to our module name ('app')
app = Flask(__name__)

# reload a persisted file we previously saves, ou Prediction Model in this case
loaded_model = joblib.load('model.pkl')


# use decorator to tell Flask witch URL should trigger our function
# we are loading index.html with default values
@app.route('/')
def index():
    """
        function display our index.html template in the user's browser
        Input str prediction_text -> is the default text displayed while waiting user input
        Input int/float -> v1 - v8 are the input variables, they have been set to default values (middle scale)
    """
    return render_template('index.html',
                           prediction_text='Change parameters to get prediction',
                           v1=3,
                           v2=90,
                           v3=90,
                           v4=50,
                           v5=95,
                           v6=30,
                           v7=0.500,
                           v8=50)


# use decorator to tell Flask witch URL should trigger our function
@app.route('/predict', methods=['POST', ])
def make_prediction():
    """
        function to take user input data, feed the model and return prediction
        we use a POST method to request data and save it to our variables
        our prediction model returns '1' for positive diagnosis and '0' for negative
        lastly, we render the html page with the current values, for better experience

        Inputs -> n/a
        Outputs str -> msg with diagnoses message
    """
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
        # the model need a DataFrame as input, without  data indexes
        user_input = pd.DataFrame(user_data, index=[0])
        [prediction] = loaded_model.predict(user_input)

    if prediction == 1:
        msg = "The prediction is positive diagnosis, patient has diabetes."
    else:
        msg = "The prediction is negative diagnosis, patient doesn't have diabetes."

    # reloads the page, but keep the current values, this way user can keep interacting with parameters
    return render_template("index.html", prediction_text=msg,
                           v1=pregnancies,
                           v2=glucose,
                           v3=blood_pressure,
                           v4=skin_thickness,
                           v5=insulin,
                           v6=BMI,
                           v7=DPF,
                           v8=age)


# starts the server
if __name__ == '__main__':
    app.run(debug=True)
