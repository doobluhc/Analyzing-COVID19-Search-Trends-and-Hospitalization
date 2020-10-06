

import pandas as pd 
import numpy as np 



# load the data into dataframes
data_search_trend_US = pd.read_csv("/Users/chengchen/Documents/GitHub/Analyzing-COVID19-Search-Trends-and-Hospitalization/2020_US_weekly_symptoms_dataset.csv")
data_hospitalization = pd.read_csv("/Users/chengchen/Documents/GitHub/Analyzing-COVID19-Search-Trends-and-Hospitalization/hospitalization_dataset.csv")




#remove symptoms that have no search data
data_search_trend_US.dropna(thresh=0, axis=1)
del data_search_trend_US['sub_region_2']
del data_search_trend_US['sub_region_2_code']
data_search_trend_US = data_search_trend_US.fillna(0)




#get the total number of regions 
regions = {}
for index,content in data_search_trend_US.iterrows():
    if content['sub_region_1_code'] not in regions:
        regions.update({content['sub_region_1_code']: 1})   



#get the hospitalization in the USA
data_hospitalization_US = data_hospitalization[data_hospitalization['open_covid_region_code'].isin(list(regions.keys()))]



#remove redundant features
data_hospitalization_US = data_hospitalization_US[['open_covid_region_code','region_name','date','hospitalized_cumulative','hospitalized_new']]



#remove regions that have too many zero entries
data_hospitalization_US = data_hospitalization_US[(data_hospitalization_US.T != 0).any()].reset_index()



#set index to date index
data_hospitalization_US['Date'] = pd.to_datetime(data_hospitalization_US['date'])
data_hospitalization_US = data_hospitalization_US.set_index('Date')
data_hospitalization_US = data_hospitalization_US.drop(['date'], axis=1)



# add a new column 
data_search_trend_US.loc[:,'hospitalized_new'] = 0




#change the value of hospitalized_new at the correct cell
for region in regions: 
    data_to_append = data_hospitalization_US[data_hospitalization_US['open_covid_region_code']==region]['hospitalized_new']
    data_to_append = data_to_append.resample('W', label='left', loffset=pd.DateOffset(days=1)).sum()
    for index,values in data_to_append.iteritems():
        if data_search_trend_US[(data_search_trend_US['date']==str(index.date()))&(data_search_trend_US['open_covid_region_code']==region)].index.values.astype(int).size==0:
            continue
        ind = data_search_trend_US[(data_search_trend_US['date']==str(index.date()))&(data_search_trend_US['open_covid_region_code']==region)].index.values.astype(int)[0]
        data_search_trend_US.at[ind,'hospitalized_new'] = values
        
    

data_search_trend_US.to_csv('/Users/chengchen/Desktop/merged_dataset.csv')

