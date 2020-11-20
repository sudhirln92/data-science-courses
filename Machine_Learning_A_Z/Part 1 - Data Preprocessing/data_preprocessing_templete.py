#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 10:41:53 2018

@author: sudhir
"""

# =============================================================================
# Data preprocessing
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
#  Importing data
# =============================================================================
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values # Independent variable
y = dataset.iloc[:,3].values # Dependent variable
#y = dataset.loc[:,'Purchased'].values # Dependent variable
dataset.head()
dataset.tail()

# =============================================================================
# Missing data
# =============================================================================
from sklearn.preprocessing import Imputer
dataset.isnull().sum() # Number of missing value
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# =============================================================================
# Encoding categorical data
# =============================================================================
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

onehotendocer = OneHotEncoder(categorical_features= [0])
X = onehotendocer.fit_transform(X).toarray()

# =============================================================================
# Splitting the dataset into the training set and test set
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=34)

# =============================================================================
# Feature Scaling
# =============================================================================
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
