#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd 
import numpy as np 


# In[23]:


# load the data into dataframes
data_search_trend_US = pd.read_csv("/Users/chengchen/Documents/GitHub/Analyzing-COVID19-Search-Trends-and-Hospitalization/2020_US_weekly_symptoms_dataset.csv")
data_hospitalization = pd.read_csv("/Users/chengchen/Documents/GitHub/Analyzing-COVID19-Search-Trends-and-Hospitalization/hospitalization_dataset.csv")


# In[24]:


#remove symptoms that have no search data
data_search_trend_US.dropna(thresh=0, axis=1)
del data_search_trend_US['sub_region_2']
del data_search_trend_US['sub_region_2_code']
data_search_trend_US = data_search_trend_US.fillna(0)


# In[25]:


#get the total number of regions 
regions = {}
for index,content in data_search_trend_US.iterrows():
    if content['sub_region_1_code'] not in regions:
        regions.update({content['sub_region_1_code']: 1})   


# In[26]:


#get the hospitalization in the USA
data_hospitalization_US = data_hospitalization[data_hospitalization['open_covid_region_code'].isin(list(regions.keys()))]


# In[27]:


#remove redundant features
data_hospitalization_US = data_hospitalization_US[['open_covid_region_code','region_name','date','hospitalized_cumulative','hospitalized_new']]


# In[28]:


#reset the index
data_hospitalization_US = data_hospitalization_US[(data_hospitalization_US.T != 0).any()].reset_index()


# In[29]:


#set index to date index
data_hospitalization_US['Date'] = pd.to_datetime(data_hospitalization_US['date'])
data_hospitalization_US = data_hospitalization_US.set_index('Date')
data_hospitalization_US = data_hospitalization_US.drop(['date'], axis=1)


# In[30]:


# add a new column 
data_search_trend_US.loc[:,'hospitalized_new'] = 0


# In[31]:


#change the value of hospitalized_new at the correct cell
for region in regions: 
    data_to_append = data_hospitalization_US[data_hospitalization_US['open_covid_region_code']==region]['hospitalized_new']
    data_to_append = data_to_append.resample('W', label='left', loffset=pd.DateOffset(days=1)).sum()
    for index,values in data_to_append.iteritems():
        if data_search_trend_US[(data_search_trend_US['date']==str(index.date()))&(data_search_trend_US['open_covid_region_code']==region)].index.values.astype(int).size==0:
            continue
        ind = data_search_trend_US[(data_search_trend_US['date']==str(index.date()))&(data_search_trend_US['open_covid_region_code']==region)].index.values.astype(int)[0]
        data_search_trend_US.at[ind,'hospitalized_new'] = values
        ß


# In[32]:


del data_search_trend_US['sub_region_1_code']
del data_search_trend_US['Unnamed: 0']


# In[34]:


data_search_trend_US = data_search_trend_US.loc[:, (data_search_trend_US==0).mean() < .7]


# In[37]:


#remove rows and cols that has over 80% of zero entries
data_search_trend_US = data_search_trend_US.loc[:, (data_search_trend_US==0).mean() < .8]
data_search_trend_US = data_search_trend_US[data_search_trend_US.astype('bool').mean(axis=1)>=0.2]


# In[39]:


data_search_trend_US.to_csv('/Users/chengchen/Desktop/merged_dataset.csv')

