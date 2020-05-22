#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 10:41:35 2018

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
# Kernel SVM classifier
# =============================================================================
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0,C=1)
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
# Applying K fold cross validation
# =============================================================================
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier,X_train,y_train,cv=10)
accuracies.mean()
accuracies.std()

# =============================================================================
# applying grid search to find best model and best parameter
# =============================================================================
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1,10,100,1000],'kernel':['linear']},
               {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.5,0.1,0.001,0.0001]}, 
               ]
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           verbose=10,
                           n_jobs =-1)
grid_search.fit(X_train,y_train)
grid_search.best_score_
grid_search.best_params_
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
    plt.title('Kernel SVM Classifier'+dataset)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()
    
plot_decision_boundary(X_train,y_train,classifier,' Training set')

plot_decision_boundary(X_test,y_test,classifier,' Testing set')

