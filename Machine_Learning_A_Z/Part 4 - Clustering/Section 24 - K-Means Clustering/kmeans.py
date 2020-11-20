#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 20:23:08 2018

@author: sudhir
"""
# =============================================================================
# Import library
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Import dataset
# =============================================================================
dataset = pd.read_csv('Mall_Customers.csv')
dataset.head()
X = dataset.iloc[:,[3,4]].values

# =============================================================================
# Using elbow method to find optimal number of cluster
# =============================================================================
from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCS')

# =============================================================================
# Applying KMeans to the mall dataset
# =============================================================================
kmeans = KMeans(n_clusters=5,random_state=0)
y_kmeans = kmeans.fit_predict(X)

# =============================================================================
# Visualising the clusters
# =============================================================================
plt.scatter(X[:,0],X[:,1],c=y_kmeans,label=set(y_kmeans),cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s =100,c='y',label='Centroid')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Cluster of clients')
plt.legend()
