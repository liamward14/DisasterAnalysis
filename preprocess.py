# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:25:52 2020

@author: liamw
"""

import time

import pandas as pd

time1 = time.time()

import seaborn as sb

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

##Imports for cloropleth map and Dash application
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

california = data[data['State']=='CA'] #Select the state of california
california.reset_index(inplace=True) #reset index


##Encode 'Declaration Type' columns with '0' or '1'
'''Disaster==0 and Emergency==1'''
label_encoder = LabelEncoder()
type_cat = california['Declaration Type']
type_encoded = label_encoder.fit_transform(type_cat)
type_DF = pd.DataFrame(type_encoded,columns=['Declaration Type'])



##Encode 'Disaster Type' column
disaster_cat = california['Disaster Type']
#Need correct column order for OneHotEncoding
cols = ['Dam/Levee Break','Drought','Earthquake','Fire','Flood','Hurricane','Other','Storm','Tsunami','Water','Winter']

disaster_encoded = label_encoder.fit_transform(disaster_cat)
auto_encoder = OneHotEncoder(categories='auto')
disaster_types_1hot = auto_encoder.fit_transform(disaster_encoded.reshape(-1,1)) #reshape as array and encode
disaster_types_1hot_mat = disaster_types_1hot.toarray() #convert to matrix
disaster_types_DF = pd.DataFrame(disaster_types_1hot_mat,columns=cols) #convert mat to df

#reset to match indicies before concatenation
disaster_types_DF.reset_index(drop=True,inplace=True)
type_DF.reset_index(drop=True, inplace=True)
california.reset_index(drop=True, inplace=True)


##Drop 'Declaration type' & 'Disaster Type' columns (they will be replaced)
california.drop(['Declaration Type','Disaster Type'],axis=1,inplace=True)

##Concatenate encoded vals
''' 
    Note that "Disaster"==0 and "Emergency"==1 in this encoding
    Note that the Disaster Types are encoded 0-9 in the order of the 'cols' list above
    but they are encoded with binary values before concatenation
'''
encoded_DF = pd.concat([california,disaster_types_DF,type_DF],axis=1,verify_integrity=True) #concat as columns
california_encoded = encoded_DF #rename


##Grab subset containing county and disaster type
county_subset = california_encoded[['County','Dam/Levee Break','Drought','Earthquake','Fire','Flood','Other','Storm','Tsunami','Water','Winter']]
county_subset.dropna(inplace=True) #remove NaN rows
county_subset_sorted = county_subset.sort_values(['County']) #sort by county
county_subset_sorted.reset_index(inplace=True,drop=True) #reset index

county_groups = california_encoded['County'].value_counts() #look at unique counties

##Extract all county names into a list
counties = []
for idx in county_groups.index:
    counties.append(idx)

##Loop through and combine sum all the values for each county
sums = {} #dict to hold summed values for each county
accounted_for = [] #this will be in the correct order to use as an index

sums_idx = 0
for index in county_subset_sorted.index:
    status = county_subset_sorted.loc[index,'County']
    if status not in accounted_for:
        subset = county_subset_sorted[county_subset_sorted['County']==status] #get row of df
        subset.drop(['County'],axis=1,inplace=True) #remove county column (to avoid repeated values in County name)
        subset_sum = dict(subset.sum()) #create a dictionary from data
        subset_sum['County'] = status #add in the county name
        sums[sums_idx] = subset_sum #add to container dictionary
        accounted_for.append(status) #tracking
        sums_idx+=1

#Note:
'''
The "sums" dictionary contains a key with the county name for its value,
as well as a key for each disaster type and its associated sum for that 
county 
'''

##Put into dataframe format for cloropleth to use:
df = pd.DataFrame(sums,index = accounted_for,columns=['Dam/Levee Break','Drought','Earthquake','Fire','Flood','Other','Storm','Tsunami','Water','Winter'])

for key, val in sums.items(): #fill df manually to get the desired form
    current_county = sums[key]['County']
    for inner_key, inner_val in sums[key].items():
        if inner_key=='County':
            pass
        else:
            df.loc[current_county,inner_key] = inner_val #fill df


##Set up cloropleth map and get california counties data (Removed because it takes more time)
# fname = r'C:\Users\liamw\OneDrive\McMasterUniversity\Machine_Learning\California\ca_counties.geojson'
# with open(fname,'r') as f:
#     geo_data = json.load(f)
#     if geo_data != np.nan:
#         print("\nFile read successfully")

##Loading the data directly from the web decreases map rendering time
link = 'https://gist.githubusercontent.com/pdbartsch/4a4ad6c68ab75d597610/raw/4adc6d41320d06e42a43e4075106932ca7b44cd5/ca_counties.geojson'
with urlopen(link) as response:
    geo_data=json.load(response)


##Extracting FIP values for formatting
features = geo_data["features"]
names = [] ##only for comparison
STATEFP = "06" #HArd coded california value
TARGET = "COUNTYFP"
COUNTYFPS = []  ##FIps codes for counties in california
for dicti in features:
    for key, val in dicti.items():
        if key=="properties":
            extraction_target = dicti[key][TARGET]
            final_val = STATEFP+extraction_target
            COUNTYFPS.append(final_val) #extracts FIPS code for each county in CA
            name = dicti[key]["NAME"]
            if 'Reservation' not in name: #Counties w/ 'Reservation' label dont need 'County' addition
                names.append(name+' County')
            else:
                names.append(name)

##Compare df and countyfps data and remove differences
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

#Reorder DataSet and create pandas object for concatenation
index = 0
fip_asst_ordered = {}
for c_name in df.index:
    for label in fip_asst.keys():
        if label==c_name:
            fip_asst_ordered[c_name] = fip_asst[label]
fip_Series = pd.Series(fip_asst_ordered)
new_df = pd.concat([df,fip_Series],axis=1,verify_integrity=True) #add FIPS column
new_df.columns = ['Dam/Levee Break','Drought','Earthquake','Fire','Flood','Other','Storm','Tsunami','Water','Winter','FIPS']

##Isolate subsets of each disaster type

dam_subset = new_df[['Dam/Levee Break','FIPS']]
dam_subset.loc[:,'FIPS'] = dam_subset.loc[:,'FIPS'].astype(str)
dam_subset.loc[:,'Dam/Levee Break'] = dam_subset.loc[:,'Dam.Levee Break'].astype(float)

flood_subset = new_df[['Flood','FIPS']]
flood_subset.loc[:,'FIPS'] = flood_subset.loc[:,'FIPS'].astype(str) #convert to string
flood_subset.loc[:,'Flood'] = flood_subset.loc[:,'Flood'].astype(float) #Convert to float for continuous color scale

##plotting with Plotly

#Use USA counties database - easier to plug-n-play with plotly choropleth
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response) #for entire country

#creat plotly figure object
fig = px.choropleth(flood_subset, geojson=counties, locations='FIPS', color='Flood',
                    color_continuous_scale="Viridis",
                    range_color=(1,11),
                    labels={'Flood':'Occurences'}
                    )

#Add title and scale layout
fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0},
                  title_text = 'Major Flood Occurences Since 1953')

fig.update_geos(fitbounds="locations", visible=False) #zoom in on California
#fig.show() #uncomment in order to see the figure

path = r'C:\Users\liamw\PycharmProjects\California\FloodMap.html'
fig.write_html(path) #save as interactive '.html'

##Produce 'Dash' app

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter

time2 = time.time()
toc = abs(time2-time1)
print("\nProcessing took {time} seconds".format(time=round(toc,3)))

