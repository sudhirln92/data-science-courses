#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 22:03:43 2018

@author: sudhir
"""
# =============================================================================
# Import library
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
#  Importing data
# =============================================================================
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:, 1:2].values
y=dataset.iloc[:,2:].values
#y = dataset.loc[:,'Purchased'].values # Dependent variable
dataset.head()
dataset.tail()

# =============================================================================
# Splitting the dataset into the training set and test set
# =============================================================================
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)
"""

# =============================================================================
# Feature Scaling
# =============================================================================
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# =============================================================================
# Fitting Simple Support Vector Regression to the Training set
# =============================================================================
from sklearn.svm import SVR
regressor = SVR(kernel='rbf',C=9.5) #
regressor.fit(X,y)

# =============================================================================
# Prediting the Training set result
# =============================================================================
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# =============================================================================
# Visualize Support Vector Regression result
# =============================================================================
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid)), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Salary vs Experience ( Support Vector Regression)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()
