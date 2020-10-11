#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 08:06:17 2020

@author: aris
Used to plot a sympton trends over time :the following code used to generate corresponding html files
"""
import os
import numpy as np
import pandas as pd
from plotly.offline import plot
import plotly.tools as tls
from datetime import datetime, timedelta
#import imgkit
#import pdfkit
#read data
dataset=pd.read_csv('./merged_dataset.csv')
dataset.head()
#select the symptom you want to plot from the dataset
Symptom=dataset[['open_covid_region_code','date','symptom:Viral pneumonia']]
#group_data=Symptom.groupby('date')

date_min=min(Symptom['date'])
#n_date=len(Symptom['date'].unique())
#draw map for the Symtom
page=[]
j=0
for i in Symptom['date'].unique():
    group1=Symptom[Symptom['date']==i]
    group1=group1.rename(columns = {'symptom:Viral pneumonia': 'DYS'})
    group1['open_covid_region_code']=group1['open_covid_region_code'].apply(lambda x:x.replace('US-', ''))
    data=[dict(type='choropleth',autocolorscale=False,locations=group1['open_covid_region_code'],
           z=group1['DYS'],locationmode='USA-states',colorscale='reds',
           colorbar=dict(title='Viral pneumonia intensity'))]
    layout=dict(title='Symp Viral pneumonia in US '+str(i),
            geo=dict(scope='usa',projection=dict(type='albers usa'),
                     showlakes=True,lakecolor='rgb(66,165,245)',),)
    fig=dict(data=data,layout=layout)
    
    page.append(plot(fig,filename='group1-VP'+str(j)+'.html',
                     auto_open=False,image_width=1280, image_height=800,image_filename='fname', 
                     image='svg'))
    j=j+1

#config = imgkit.configuration(wkhtmltopdf="C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltoimage.exe")
#pdfkit.from_file('123.html', 'out.pdf',configuration=config)
#imgkit.from_file('123.html', 'out.jpg')
'''
After the code,just download the png files for each html file and then use the following cmd to generate the video"
ffmpeg -r 1 -f image2 -pattern_type glob -i "*?png" -vcodec libx264 -crf 20 -pix_fmt yuv420p output.mp4
'''









