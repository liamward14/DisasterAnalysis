# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:25:52 2020

@author: liamw
"""

import time

import pandas as pd
import numpy as np

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
california_original = data[data['State']=='CA']
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
california_encoded.drop(['index'],axis=1,inplace=True) #remove 'index' column (result of resetting index)

##Seperate Dates and label
labelled_dates_df = pd.DataFrame(california_encoded[['Declaration Number','Declaration Date']]).astype(str)
years = []
for date in labelled_dates_df['Declaration Date']:
    split_text = date.split("/")
    row_year = split_text[2]
    years.append(row_year)
years_df = pd.DataFrame(years,columns=['Year'])
year_only_df = pd.concat([california_encoded.drop(['Start Date','End Date','Close Date'],axis=1),years_df],axis=1,verify_integrity=True)
##Index reset
california_original.reset_index(inplace=True)
year_only_nonencoded_df = pd.concat([california_original.drop(['Start Date','End Date','Close Date'],axis=1),years_df],axis=1,verify_integrity=True)


##Isolate 2017 data
vals_2017 = year_only_nonencoded_df[year_only_nonencoded_df['Year']=='2017']
drop_idx = vals_2017[vals_2017['County']=='Hoopa Valley Indian Reservation'].index.values[0]
vals_2017.drop(drop_idx,inplace=True)



# df_2017 = labelled_years_df[labelled_years_df['Year']=='2017']
# df_2017.sort_values('Declaration Number',inplace=True)
#
# encoded_2017 = california_encoded[california_encoded['Year']=='2017']
# print(encoded_2017)
#
# #Organize 2017 data
# vals_2017 = {}
# for index in df_2017.index:
#     entry_data = dict(california_encoded.loc[index])
#     label = entry_data['Declaration Number']
#     vals_2017[label] = entry_data



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
for name in names: #b/c they are in the same order
    fip_asst[name] = FIPS[index]
    index+=1

#Reorder DataSet and create pandas object for concatenation
index = 0
fip_asst_ordered = {} #gives FIPS code associated with each county
for c_name in df.index:
    for label in fip_asst.keys():
        if label==c_name:
            fip_asst_ordered[c_name] = fip_asst[label]
fip_Series = pd.Series(fip_asst_ordered)
new_df = pd.concat([df,fip_Series],axis=1,verify_integrity=True) #add FIPS column
new_df.columns = ['Dam/Levee Break','Drought','Earthquake','Fire','Flood','Other','Storm','Tsunami','Water','Winter','FIPS']

##Isolate subsets of each disaster type and ensure correct dtype for plotly object

dam_subset = new_df[['Dam/Levee Break','FIPS']]
dam_subset.loc[:,'FIPS'] = dam_subset.loc[:,'FIPS'].astype(str)
dam_subset.loc[:,'Dam/Levee Break'] = dam_subset.loc[:,'Dam/Levee Break'].astype(float)

drought_subset = new_df[['Drought','FIPS']]
drought_subset.loc[:,'FIPS'] = drought_subset.loc[:,'FIPS'].astype(str)
drought_subset.loc[:,'Drought'] = drought_subset.loc[:,'Drought'].astype(float)

quake_subset = new_df[['Earthquake','FIPS']]
quake_subset.loc[:,'FIPS'] = quake_subset.loc[:,'FIPS'].astype(str)
quake_subset.loc[:,'Earthquake'] = quake_subset.loc[:,'Earthquake'].astype(float)

fire_subset = new_df[['Fire','FIPS']]
fire_subset.loc[:,'FIPS'] = fire_subset.loc[:,'FIPS'].astype(str)
fire_subset.loc[:,'Fire'] = fire_subset.loc[:,'Fire'].astype(float)

flood_subset = new_df[['Flood','FIPS']]
flood_subset.loc[:,'FIPS'] = flood_subset.loc[:,'FIPS'].astype(str) #convert to string
flood_subset.loc[:,'Flood'] = flood_subset.loc[:,'Flood'].astype(float) #Convert to float for continuous color scale
flood_range = (1,11)

##Ignore 'Other' classification

storm_subset = new_df[['Storm','FIPS']]
storm_subset.loc[:,'FIPS'] = storm_subset.loc[:,'FIPS'].astype(str)
storm_subset.loc[:,'Storm'] = storm_subset.loc[:,'Storm'].astype(float)
storm_range = (1,8)

tsunami_subset = new_df[['Tsunami','FIPS']]
tsunami_subset.loc[:,'FIPS'] = tsunami_subset.loc[:,'FIPS'].astype(str)
tsunami_subset.loc[:,'Tsunami'] = tsunami_subset.loc[:,'Tsunami'].astype(float)

water_subset = new_df[['Water','FIPS']]
water_subset.loc[:,'FIPS'] = water_subset.loc[:,'FIPS'].astype(str)
water_subset.loc[:,'Water'] = water_subset.loc[:,'Water'].astype(float)

winter_subset = new_df[['Winter','FIPS']]
winter_subset.loc[:,'FIPS'] = winter_subset.loc[:,'FIPS'].astype(str)
winter_subset.loc[:,'Winter'] = winter_subset.loc[:,'Winter'].astype(float)

##Create an ordered FIPS Series and append with vals_2017 (see line 92)
fips_2017 = {}
dict_ind = 0
for index in vals_2017.index:
    prohibited = 'Reservation'
    county_temp = vals_2017.loc[index,'County']
    if prohibited in county_temp:
        pass
    else:
        fips_temp = fip_asst_ordered[county_temp]
        fips_2017[dict_ind] = str(fips_temp)
        dict_ind+=1


fips_2017_df = pd.DataFrame(fips_2017,columns=['FIPS'])
for key,val in fips_2017.items():
    fips_2017_df.loc[key] = val #fill FIPS DF - not sure why it isnt working

cols_nonencoded = ['Declaration Number','Declaration Type','Declaration Date','State','County',
               'Disaster Type','Disaster Title','Individual Assistance Program',
               'Individuals & Households Program',
               'Public Assistance Program','Hazard Mitigation Program',
               'Year','FIPS']
final_2017 = pd.concat([vals_2017.reset_index(),fips_2017_df],axis=1,verify_integrity=True)
final_2017.drop(['level_0','index'],inplace=True,axis=1)

for cty in list(fip_asst_ordered.keys()): #add counties that are not in that year to fill map
    if cty in list(final_2017['County'].values):
        pass
    else:
        row = ['Arb Number','Dec Type','Date','CA',cty,'None','Title','IAP','IHP','PAP','HMP','2017',fip_asst_ordered[cty]]
        new_row = {}
        idx = 0
        for key in cols_nonencoded: #fill dictionary with keys as columns to append to df
            new_row[key] = row[idx]
            idx+=1
        final_2017 = final_2017.append(new_row,ignore_index=True)


print(final_2017.head())
print(final_2017.info())


##Plotting with plotly for 2017 data
'''
In order to get this to work properly, need to take care
of counties that have two or more disaster types for 1 year
'''
#Use USA counties database - easier to plug-n-play with plotly choropleth
# with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
#     counties = json.load(response) #for entire country
#
# num_plots = len(new_df.columns)-1
#
# #creat plotly figure object
# fig = px.choropleth(final_2017.drop([5,32,36]), geojson=counties, locations='FIPS', color='Disaster Type',
#                     labels='Disaster Type'
#                     )
# #the drop above ^ is a temporary fix for multiple values
#
# #Add title and scale layout
# fig.update_layout(margin={"r":0,"t":40,"l":0,"b":40},
#                   height=720,
#                   title_text = 'Major Disasters in 2017')
#
# fig.update_geos(fitbounds="locations",
#                 visible=False,
#                 showsubunits=True) #zoom in on California
# #fig.show() #uncomment in order to see the figure
#
# path = r'C:\Users\liamw\PycharmProjects\California\Disaster2017Map.html'
# fig.write_html(path) #save as interactive '.html'
#
# ##Produce 'Dash' app
#
# app = dash.Dash()
# app.layout = html.Div([
#     dcc.Graph(figure=fig)
# ])
#
# app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter

# print(fip_asst_ordered)
# ##Create 2017 dataset for plotting
# for key,val in vals_2017.items():
#     instance = vals_2017[key]
#     for key_inner, val_inner in instance.items():
#         if key_inner=='County':
#            #what is the fips code for this value?
#             county = val_inner
#           #  fips_code = fip_asst_ordered[county]
# print(vals_2017)

##plotting with Plotly

#Use USA counties database - easier to plug-n-play with plotly choropleth
# with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
#     counties = json.load(response) #for entire country
#
# num_plots = len(new_df.columns)-1
#
# #creat plotly figure object
# fig = px.choropleth(storm_subset, geojson=counties, locations='FIPS', color='Storm',
#                     color_continuous_scale="Viridis",
#                     range_color=storm_range,
#                     labels={'Storm':'Occurences'}
#                     )
#
# #Add title and scale layout
# fig.update_layout(margin={"r":0,"t":40,"l":0,"b":40},
#                   height=720,
#                   title_text = 'Major Storm Occurences Since 1953')
#
# fig.update_geos(fitbounds="locations", visible=False) #zoom in on California
# #fig.show() #uncomment in order to see the figure
#
# path = r'C:\Users\liamw\PycharmProjects\California\StormMap.html'
# fig.write_html(path) #save as interactive '.html'
#
# ##Produce 'Dash' app
#
# app = dash.Dash()
# app.layout = html.Div([
#     dcc.Graph(figure=fig)
# ])
#
# app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter
#
# time2 = time.time()
# toc = abs(time2-time1)
# print("Processing took {time} seconds".format(time=round(toc,3)))
