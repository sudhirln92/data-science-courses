#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 11:24:29 2018

@author: sudhir
"""

# =============================================================================
# 
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Importing data
# =============================================================================
dataset = pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values
dataset.head()
dataset.tail()

# =============================================================================
# Splitting the dataset into the training set and test set
# =============================================================================
"""from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)"""

# =============================================================================
# Fitting Linear regression to the dataset
# =============================================================================
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)

# =============================================================================
# Fitting Polynomial Regression to the dataset
# =============================================================================
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# =============================================================================
# Visualize Linear regression result
# =============================================================================
plt.plot(X,y,'r.')
plt.plot(X,regressor.predict(X),'b-')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('True or Bluff (Linear Regression)')

# =============================================================================
# Visualize Polynomial regression result
# =============================================================================
plt.plot(X,y,'r.')
plt.plot(X,lin_reg_2.predict(X_poly),'b--')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('True or Bluff (Polynomial Regression)')

# =============================================================================
# Predicting a new result with Linear Regression  
# =============================================================================
regressor.predict(6.5)

# =============================================================================
# Predinting a new result with Polynomial Regression
# =============================================================================
lin_reg_2.predict(poly_reg.fit_transform(6.5))
