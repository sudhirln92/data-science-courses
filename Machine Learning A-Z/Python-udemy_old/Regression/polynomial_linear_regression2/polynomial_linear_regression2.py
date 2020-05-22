#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

##Spliting dataset into 
#from sklearn.cross_validation import train_test_split
#X_train,X_test, y_train, y_test=train_test_split(X,y,test_size=.2, random_state=0)

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X, y)

# Fiting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =2)
X_poly = poly_reg.fit_transform(X)