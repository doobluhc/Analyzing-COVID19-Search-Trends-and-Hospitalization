#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


# In[2]:


df = pd.read_csv('/Users/chengchen/Documents/GitHub/Analyzing-COVID19-Search-Trends-and-Hospitalization/merged_dataset.csv')


# In[3]:


df = df.sample(frac = 1) 


# In[4]:


X = df.drop(columns=['Unnamed: 0','open_covid_region_code','country_region_code',
                    'country_region_code','country_region','sub_region_1','date','hospitalized_new'])


# In[5]:


y = df['hospitalized_new'].values


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[7]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[8]:


#cross validation
tree_regressor_cv = DecisionTreeRegressor(criterion='mse')
param_grid = {'max_depth': np.arange(5, 30)}
tree_gscv = GridSearchCV(tree_regressor_cv, param_grid, cv=5)
tree_gscv.fit(X, y)


# In[9]:


#visulize the cross validation
import matplotlib.pyplot as plt 
plot_y = tree_gscv.cv_results_['mean_test_score']
plot_x = np.arange(5, 30)
plt.plot(plot_x,plot_y) 
plt.xlabel('max_depth')
plt.ylabel('test score')
plt.show()


# In[10]:


tree_gscv.best_params_['max_depth']


# In[11]:


#train the model
best_depth = tree_gscv.best_params_['max_depth']
tree_regressor = DecisionTreeRegressor(max_depth=best_depth)
tree_regressor.fit(X_train,y_train)


# In[12]:


#test the model
tree_regressor.score(X_test, y_test)

