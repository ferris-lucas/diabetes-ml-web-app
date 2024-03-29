# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wHOgY0Kjc9IA2I7ELOAwqrJ-Dc2M4Ry-
"""

#Description: this program detects if someone has diabetes using machine learning and python!
#Import the libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

#Get the data for the model
df = pd.read_csv('https://raw.githubusercontent.com/ferris77/ml-web-app/main/diabetes.csv')

#We need to preprocess the data replacing zero values with suitable values (means)
df['Glucose'] = np.where(df['Glucose']==0,df['Glucose'].mean(),df['Glucose'])
df['BloodPressure'] = np.where(df['BloodPressure']==0,df['BloodPressure'].mean(),df['BloodPressure'])
df['SkinThickness'] = np.where(df['SkinThickness']==0,df['SkinThickness'].median(),df['SkinThickness'])
df['Insulin'] = np.where(df['Insulin']==0,df['Insulin'].median(),df['Insulin'])
df['BMI'] = np.where(df['BMI']==0,df['BMI'].mean(),df['BMI'])

#Split the data into independentent 'X' and dependente 'Y' variables
X = df.iloc[:, 0:8].values #we want the array, not the df
Y = df.iloc[:, -1].values

#Split the data into 75% training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

#Create and train the model
model = RandomForestClassifier()
model.fit(X_train, Y_train)

#Let's check our models accuracy
predictions = model.predict(X_test)
print(f'Our first RFC model has an accuracy of: {accuracy_score(Y_test, predictions)*100:.2f}%')

# Manual Hyperparameter Tuning
manual_tuned_model = RandomForestClassifier(n_estimators=100,criterion='gini',
                                            max_features='sqrt',
                                            min_samples_leaf=5,random_state=0)
manual_tuned_model.fit(X_train, Y_train)
manual_tuned_predictions = manual_tuned_model.predict(X_test)
print(f'Our manual tunned RFC model has an accuracy of: {accuracy_score(Y_test, manual_tuned_predictions)*100:.2f}%')

#Hypertuning
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 1000,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,14]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,6,8]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}

print(random_grid)

auto_tuned_model = RandomForestClassifier()
RFC_randomcv = RandomizedSearchCV(estimator = auto_tuned_model,
                                  param_distributions = random_grid,
                                  n_iter = 20,
                                  cv = 3,
                                  verbose = 2,
                                  random_state = 100,
                                  n_jobs = -1)
### fit the randomized model
RFC_randomcv.fit(X_train,Y_train)

RFC_randomcv.best_params_

best_random_grid = RFC_randomcv.best_estimator_

auto_tuned_predictions = best_random_grid.predict(X_test)
print(f'Our manual tunned RFC model has an accuracy of: {accuracy_score(Y_test, auto_tuned_predictions)*100:.2f}%')

print(f'model accuracy: {accuracy_score(Y_test, predictions)*100:.2f}%')
print(f'manual_tuned_model accuracy: {accuracy_score(Y_test, manual_tuned_predictions)*100:.2f}%')
print(f'auto_tuned_model accuracy: {accuracy_score(Y_test, auto_tuned_predictions)*100:.2f}%')

#We reached higher accuracy with manual-tunned model, we will select this model
#to serialize and save to disk
joblib.dump(manual_tuned_model, 'model.pkl')