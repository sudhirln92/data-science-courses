#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 11:14:33 2018

@author: sudhir
"""
# =============================================================================
# Import library
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Import library
# =============================================================================
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

# =============================================================================
# Encoding categorical variable
# =============================================================================
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:,1] = labelencoder_X1.fit_transform(X[:,1])
labelencoder_X2 = LabelEncoder()
X[:,2] = labelencoder_X2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features= [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# =============================================================================
# Splitting the dataset into training and test set
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

# =============================================================================
# Fitting XGBoost to the training set
# =============================================================================
import xgboost as xgb
classifier = xgb.XGBClassifier()
classifier.fit(X_train,y_train)

# =============================================================================
# Prediction
# =============================================================================
y_pred = classifier.predict(X_test)

# =============================================================================
# Making the confusion matrix
# =============================================================================
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
