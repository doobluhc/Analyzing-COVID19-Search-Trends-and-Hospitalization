#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

#%%
#import data
dataset=pd.read_csv('./merged_dataset.csv')
data_t1=dataset.loc[:,'symptom:Adrenal crisis':'symptom:Viral pneumonia']

cols=list(data_t1.columns)
symptoms=[]
for s in cols:
    if 'symptom:' in s:
        symptoms.append(s)        
X=data_t1
X=data_t1.loc[:,symptoms].values
X=StandardScaler().fit_transform(X)

pca=PCA(n_components=2)
pca.fit(X)
X_reduced=pca.transform(X)

plt.scatter(X_reduced[:,0],X_reduced[:,1],cmap=plt.cm.get_cmap('viridis',3),s=4)
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=15)
plt.ylabel('Principal Component - 2',fontsize=15)
plt.title("Principal Component Analysis of Symptoms Dataset",fontsize=20)

#%%
# Clustering for high-dimensional KMeans
kmeans_high=KMeans(n_clusters=3,random_state=0)
kmeans_high.fit(X)
y_pred_high=kmeans_high.predict(X)

kmeans_low=KMeans(n_clusters=3,random_state=0)
kmeans_low.fit(X_reduced)
y_pred_low=kmeans_low.predict(X_reduced)



plt.subplot(2,1,1)
plt.scatter(X_reduced[:,0],X_reduced[:,1],c=y_pred_high,
            cmap=plt.cm.get_cmap('viridis',3), s=4)
plt.colorbar(ticks=[0,1,2])
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=10)
plt.ylabel('Principal Component - 2',fontsize=10)
plt.title("Cluster labels for high-dimensional KMeans",fontsize=12)



plt.subplot(2,1,2)
plt.scatter(X_reduced[:,0],X_reduced[:,1],c=y_pred_low,
            cmap=plt.cm.get_cmap('viridis',3), s=4)
plt.colorbar(ticks=[0,1,2])
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=10)
plt.ylabel('Principal Component - 2',fontsize=10)
plt.title("Cluster labels for low-dimensional KMeans",fontsize=12)







