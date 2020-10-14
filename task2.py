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

#cols=list(data_t1.columns)
#symptoms=[]
#for s in cols:
#    if 'symptom:' in s:
#        symptoms.append(s)        
X=data_t1
#X=data_t1.loc[:,symptoms].values
#X=StandardScaler().fit_transform(X)

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
k_val = list(range(2,19,1))

fig,a=plt.subplots(17,2,figsize=(20,20))
for i,k in enumerate(k_val):
    km_raw=KMeans(n_clusters=k,random_state=0)
    km_raw.fit(X)
    y_pred_raw=km_raw.predict(X)
    
    a[i][0].scatter(X_reduced[:,0],X_reduced[:,1],c=y_pred_raw,cmap=plt.cm.get_cmap('tab20'),s=1)
    a[i][0].set_title('Raw data: '+f'K={k}')
    a[i][0].set_xlabel("PC-1")
    a[i][0].set_ylabel("PC-2")
    
    km_raw=KMeans(n_clusters=k,random_state=0)
    km_raw.fit(X_reduced)
    y_pred_raw=km_raw.predict(X_reduced)
    
    a[i][1].scatter(X_reduced[:,0],X_reduced[:,1],c=y_pred_raw,cmap=plt.cm.get_cmap('tab20'),s=1)
    a[i][1].set_title('Reduced data: '+f'K={k}')
    a[i][1].set_xlabel("PC-1")
    a[i][1].set_ylabel("PC-2")

