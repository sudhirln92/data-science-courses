#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 08:57:03 2018

@author: sudhir
"""
# =============================================================================
# 
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
#  Importing data
# =============================================================================
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:,1].values
#y = dataset.loc[:,'Purchased'].values # Dependent variable
dataset.head()
dataset.tail()

# =============================================================================
# Splitting the dataset into the training set and test set
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)

# =============================================================================
# Fitting Simple Linear Regression to the Training set
# =============================================================================
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# =============================================================================
# Prediting the Training set result
# =============================================================================
y_pred = regressor.predict(X_test)

# =============================================================================
# Visualize Training set result
# =============================================================================
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

# =============================================================================
# Visualize Training set result
# =============================================================================
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

