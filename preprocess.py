# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:25:52 2020

@author: liamw
"""

import numpy as np
import pandas as pd
import time

time1 = time.time()

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sb


import sklearn
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

##Imports for cloropleth map
from urllib.request import urlopen
import json
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html

sb.set_style('whitegrid')

##Read Data
address = r'C:\Users\liamw\OneDrive\McMasterUniversity\Machine_Learning\California\database.csv'
data = pd.read_csv(address)
data.columns = ['Declaration Number','Declaration Type','Declaration Date','State','County',
               'Disaster Type','Disaster Title','Start Date','End Date','Close Date',
               'Individual Assistance Program','Individuals & Households Program',
               'Public Assistance Program','Hazard Mitigation Program']

california = data[data['State']=='CA']
california.reset_index(inplace=True)


##Encode 'Declaration Type' columns with '0' or '1'
label_encoder = LabelEncoder()
type_cat = california['Declaration Type']
type_encoded = label_encoder.fit_transform(type_cat)
type_DF = pd.DataFrame(type_encoded,columns=['Declaration Type'])

type_DF.reset_index(drop=True, inplace=True)
california.reset_index(drop=True, inplace=True)


##Encode 'Disaster Type' column
disaster_cat = california['Disaster Type']
cols = ['Dam/Levee Break','Drought','Earthquake','Fire','Flood','Hurricane','Other','Storm','Tsunami','Water','Winter']
        
disaster_encoded = label_encoder.fit_transform(disaster_cat)
auto_encoder = OneHotEncoder(categories='auto')
disaster_types_1hot = auto_encoder.fit_transform(disaster_encoded.reshape(-1,1))
disaster_types_1hot_mat = disaster_types_1hot.toarray()
disaster_types_DF = pd.DataFrame(disaster_types_1hot_mat,columns=cols)
disaster_types_DF.reset_index(drop=True,inplace=True)

##Drop 'Declaration type' & 'Disaster Type' columns
california.drop(['Declaration Type','Disaster Type'],axis=1,inplace=True)

##Concatenate encoded vals
''' Note that "Disaster"==0 and "Emergency"==1 in this encoding
    Note that the Disaster Types are encoded 0-9 in the order of the 'cols' list above'''
encoded_DF = pd.concat([california,disaster_types_DF,type_DF],axis=1,verify_integrity=True)
california_encoded = encoded_DF #rename


##GRab subset containing county and disaster type
county_subset = california_encoded[['County','Dam/Levee Break','Drought','Earthquake','Fire','Flood','Other','Storm','Tsunami','Water','Winter']]
county_subset.dropna(inplace=True)
county_subset.reset_index(inplace=True,drop=True)
#print(county_subset.head())
county_subset_sorted = county_subset.sort_values(['County'])
county_subset_sorted.reset_index(inplace=True,drop=True)
#print(county_subset_sorted.head())
county_groups = california_encoded['County'].value_counts() #look at unique counties

##Extract all county names into a list
counties = []
for idx in county_groups.index:
    counties.append(idx)

#print(county_subset_sorted[county_subset_sorted['County']=='Alameda County'])

##Loop through and combine sum all the values for each county
sums = {} #dict to hold summed values for each county
accounted_for = [] #this will be in the correct order to use as an index

sums_idx = 0
for index in county_subset_sorted.index:
    status = county_subset_sorted.loc[index,'County']
    if status not in accounted_for:
        subset = county_subset_sorted[county_subset_sorted['County']==status]
        subset.drop(['County'],axis=1,inplace=True)
        subset_sum = dict(subset.sum())
        subset_sum['County'] = status
        sums[sums_idx] = subset_sum
        accounted_for.append(status)
        sums_idx+=1

#Note:
'''The "sums" dictionary contains a key with the county name for its value,
as well as a key for each disaster type and its associated sum for that 
county '''

##Put into dataframe format for cloropleth to use:
df = pd.DataFrame(sums,index = accounted_for,columns=['Dam/Levee Break','Drought','Earthquake','Fire','Flood','Other','Storm','Tsunami','Water','Winter'])

for key, val in sums.items(): #fill df manually
    current_county = sums[key]['County']
    for inner_key, inner_val in sums[key].items():
        if inner_key=='County':
            pass
        else:
            df.loc[current_county,inner_key] = inner_val
    



##Plot common disaster in the LA county area
#print(county_groups)
# LA = county_subset[county_subset['County']=='Los Angeles County']
# LA.drop(['County'],axis=1,inplace=True)
# LA.reset_index(inplace=True,drop=True)
# LA_sum = LA.sum(axis=0)
# LA_sum.drop(['Other'],axis=0,inplace=True)

# ##Remove zeros
# labels_rem = []
# explode = []
# for index in LA_sum.index:
#     status = LA_sum[index]
#     if status==0.0:
#         labels_rem.append(index)
#         LA_sum.drop(index,inplace=True)
#     else:
#         explode.append(0.2)
     
# print(LA_sum)
# base_labels = ['Dam/Levee Break','Drought','Earthquake','Fire','Flood','Storm','Tsunami','Water','Winter']
# for label in labels_rem:
#     if label in base_labels:
#         base_labels.remove(label)


# ##Visualize with seaborn 
# colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','aqua']
# plt.pie(LA_sum,labels=base_labels,shadow=True,explode=explode,colors=colors)
# plt.title('Most Prevelant Los Angeles County Disasters since 1953')


##Set up cloropleth map and get california counties data
# fname = r'C:\Users\liamw\OneDrive\McMasterUniversity\Machine_Learning\California\ca_counties.geojson'
# with open(fname,'r') as f:
#     geo_data = json.load(f)
#     if geo_data != np.nan:
#         print("\nFile read successfully")
link = 'https://gist.githubusercontent.com/pdbartsch/4a4ad6c68ab75d597610/raw/4adc6d41320d06e42a43e4075106932ca7b44cd5/ca_counties.geojson'
with urlopen(link) as response:
    geo_data=json.load(response)



##Extracting FIP values for formatting
features = geo_data["features"]
names = [] ##only for comparison
STATEFP = "06"
TARGET = "COUNTYFP"
COUNTYFPS = []  ##FIps codes for counties in california
for dicti in features:
    for key, val in dicti.items():
        if key=="properties":
            extraction_target = dicti[key][TARGET]
            final_val = STATEFP+extraction_target
            COUNTYFPS.append(final_val)
            name = dicti[key]["NAME"]
            if 'Reservation' not in name:
                names.append(name+' County')
            else:
                names.append(name)

##Compare df and countyfps data:
outliers = []
buffer = []
df_indicies = df.index
for index in df_indicies:
    buffer.append(index)
    
for name in buffer:
    if name not in names:
        outliers.append(name)

for value in outliers:
    df.drop(value,inplace=True) ##get rid of reservation data
        
##Add FIPS column to df
FIPS = pd.Series(COUNTYFPS)
fip_asst = {} #contains associate names of fips codes
index = 0
for name in names:
    fip_asst[name] = FIPS[index]
    index+=1

#Loop through and compare names
index = 0
fip_asst_ordered = {}
for c_name in df.index:
    for label in fip_asst.keys():
        if label==c_name:
            fip_asst_ordered[c_name] = fip_asst[label]
fip_Series = pd.Series(fip_asst_ordered)
new_df = pd.concat([df,fip_Series],axis=1,verify_integrity=True)
new_df.columns = ['Dam/Levee Break','Drought','Earthquake','Fire','Flood','Other','Storm','Tsunami','Water','Winter','FIPS']

flood_subset = new_df[['Flood','FIPS']]
flood_subset['FIPS'] = flood_subset['FIPS'].astype(str)
flood_subset['Flood'] = flood_subset['Flood'].astype(float)

##plotting with Plotly

#tls.set_credentials_file(username='wlw1',api_key='aUKcWEG8zmxpkwUvzUJ1')
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

fig = px.choropleth(flood_subset, geojson=counties, locations='FIPS', color='Flood',
                    color_continuous_scale="Viridis",
                    range_color=(1,11),
                    labels={'Flood':'Occurences'}
                    )
fig.update_layout(title_text = 'Major Flood Occurences Since 1953',
                  margin={"r":0,"t":0,"l":0,"b":0})
fig.update_geos(fitbounds="locations", visible=False)
fig.show()
path = r'C:\Users\liamw\PycharmProjects\California\FloodMap.html'
fig.write_html(path)

##Uncomment to produce 'Dash' app
'''
app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter
'''






time2 = time.time()
toc = abs(time2-time1)
print("\nProcessing took {time} seconds".format(time=round(toc,3)))

