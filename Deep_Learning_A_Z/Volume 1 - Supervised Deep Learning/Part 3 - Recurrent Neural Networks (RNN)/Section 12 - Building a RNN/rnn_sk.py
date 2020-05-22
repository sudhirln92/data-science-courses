#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 09:28:56 2018

@author: sudhir
"""
# =============================================================================
# Recurrent neural network
# Part1 - Data preprocessing
# Importing the library
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Importing the training set
# =============================================================================
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:,1:2].values

# =============================================================================
# Feature scaling
# =============================================================================
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# =============================================================================
# Getting the input and the outputs
# =============================================================================
X_train = training_set[0:1257]
y_train = training_set[1:1258]

# Reshape
X_train = np.reshape(X_train, (1257, 1, 1))

# =============================================================================
# Part2 Building the RNN
# Importing the keras libraries and package
# =============================================================================
from keras.models import Sequential
from keras.layers import Dense, LSTM

#Intialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units= 4, activation='sigmoid', input_shape=(None, 1)))

# Adding the output layer
regressor.add(Dense(units= 1,))

# Compiling the RNN
regressor.compile(optimizer='adam', loss ='mean_squared_error')

# Fitting the RNN to the training set
regressor.fit(X_train, y_train, batch_size= 32, epochs= 200)

# =============================================================================
# Part3 - Making the prediction and visualising the results
# Getting the real stock price of 2017
# =============================================================================
test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values

inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20, 1,1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# =============================================================================
# Visualizing the result
# =============================================================================
plt.plot(real_stock_price, color='red', label = 'Real')
plt.plot(predicted_stock_price, color = 'green', label='Predicted')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()

# =============================================================================
# Home work
# Getting the real stock price of 2012 - 2016
# =============================================================================
real_stock_price_train = pd.read_csv('Google_Stock_Price_Train.csv')
real_stock_price_train = real_stock_price_train.iloc[:,1:2].values

# Getting the predicted stock price of 2012 2016
predicted_stock_price_train = regressor.predict(X_train)
predicted_stock_price_train = sc.inverse_transform(predicted_stock_price_train)

# =============================================================================
# Visualizing the result
# =============================================================================
plt.plot(real_stock_price_train, color='red', label = 'Real')
plt.plot(predicted_stock_price_train, color = 'green', label='Predicted')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()

# =============================================================================
# Part4 Evaluating the RNN
# =============================================================================
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(real_stock_price, predicted_stock_price) ** 0.5







