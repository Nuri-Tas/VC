import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

dataset = pd.read_csv('C:/Users/Lenovo/Desktop/TSA_Kaggle/tunnel.csv')
dataset['lag'] = dataset.NumVehicles.shift(1)

X = dataset.loc[1:, ['lag']].values
y = dataset.loc[1:, ['NumVehicles']].values
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
# Fitting Simple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test)
# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))