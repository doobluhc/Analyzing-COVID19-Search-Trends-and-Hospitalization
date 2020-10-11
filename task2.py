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

#import data
dataset=pd.read_csv('./merged_dataset.csv')
data_t1=dataset.loc[:,'symptom:Adrenal crisis':'symptom:Yawn']

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
