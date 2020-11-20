#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:50:51 2018

@author: sudhir
"""
# =============================================================================
# Import library
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Import data set
# =============================================================================
df = pd.read_csv('Social_Network_Ads.csv')
df.head()
X = df.iloc[:,[2,3]].values
y = df.iloc[:,4].values

# =============================================================================
# Train Test Split
# =============================================================================
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=0)

# =============================================================================
# Feature Scaling
# =============================================================================
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =============================================================================
# K Nearest Neighbor classifier
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(X_train,y_train)

# =============================================================================
# Prediction
# =============================================================================
y_pred = classifier.predict(X_test)

# =============================================================================
# Making confusion matrix
# =============================================================================
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# =============================================================================
# Visualize training and testing set result
# =============================================================================
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X_set,y_set,classifier,dataset):
    X1,X2 = np.meshgrid(np.arange(X_set[:,0].min()-1,X_set[:,0].max()+1, step=0.01),
                        np.arange(X_set[:,1].min()-1,X_set[:,1].max()+1, step=0.01))
    X_new = np.c_[X1.ravel(),X2.ravel()]
    pred = classifier.predict(X_new).reshape(X1.shape)
    plt.contourf(X1,X2,pred, alpha=0.2, cmap=ListedColormap(('red','green')))
    
    plt.scatter(X_set[:,0][y_set==0], X_set[:,1][y_set==0], c='r', s=10, label='0')
    plt.scatter(X_set[:,0][y_set==1], X_set[:,1][y_set==1], c='g', s=10,label='1')
    plt.xlim(X1.min(),X1.max())
    plt.ylim(X2.min(),X2.max())
    plt.title('KNN Classifier'+dataset)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()
    
plot_decision_boundary(X_train,y_train,classifier,' Training set')

plot_decision_boundary(X_test,y_test,classifier,' Testing set')

