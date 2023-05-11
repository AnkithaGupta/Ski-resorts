#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 23:33:19 2023

@author: aakshasequeira
"""

## -------------------------------------PREPROCESSING------------------------------------ ##
import pandas as pd


#Import airport data
airports = pd.read_excel("airports.xls")

#Only keeping relevant columns for analysis
airports = airports.loc[:, ['airport_id', 'latitude', 'longitude', 'name', 'city', 'country']]

#Rename columns 
airports = airports.rename(columns={'airport_id':'Airport_ID', 'latitude':'Latitude', 'longitude':'Longitude', 'name':'Airport_Name', 'city':'Airport_City', 'country':'Airport_Country'})


#Import resort data
resorts = pd.read_excel("resorts.xls")

#Rename Columns
resorts = resorts.rename(columns={'Country':'Resort_Country'})

#Matching LAT and LONG from each dataset
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Create a numpy array with Latitude and Longitude values from Airports dataframe
X = np.array(airports[['Latitude', 'Longitude']])

# Fit a nearest neighbor model on the points in Airports dataframe
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)

# Get the indices of the nearest point in Resorts dataframe for each point in Airports dataframe
distances, indices = nbrs.kneighbors(resorts[['Latitude', 'Longitude']])

# Create a new column in Resorts dataframe with the index of the nearest point in Airports
resorts['nearest_index'] = indices.flatten()

# Merge Resorts with Airports using the nearest_index column
merged_df = pd.merge(resorts, airports, left_on='nearest_index', right_index=True)

## Modeling
#Creating Dummies
merged_df[['Child friendly', 'Snowparks', 'Nightskiing', 'Summer skiing']] = merged_df[['Child friendly', 'Snowparks', 'Nightskiing', 'Summer skiing']].replace({'Yes': 1, 'No': 0})

#Calculating distance between airports and resorts

from geopy.distance import geodesic

# define a function to calculate distance between two coordinates
def calc_distance(row):
    coords_1 = (row['Latitude_x'], row['Longitude_x'])
    coords_2 = (row['Latitude_y'], row['Longitude_y'])
    return geodesic(coords_1, coords_2).miles

# apply the function to each row of the DataFrame
merged_df['distance_miles'] = merged_df.apply(calc_distance, axis=1)

## -------------------------------------EDA------------------------------------ ##
## Number of Ski Resorts by Country

import plotly.graph_objs as go

airport_counts = merged_df.groupby("Airport_Country")["Airport_ID"].nunique()

# Sort the airport counts in descending order
airport_counts = airport_counts.sort_values(ascending=False)

# Create a bar chart trace for the airport counts
airport_trace = go.Bar(
    y=airport_counts.index,
    x=airport_counts.values,
    orientation='h',
    marker=dict(
        color=airport_counts.values,
        colorscale='Viridis',
        reversescale=True
    ),
)

# Create the layout for the chart
airport_layout = go.Layout(
    title='Number of Airports by Country',
    xaxis=dict(title='Count'),
    yaxis=dict(title='Country'),
)

# Create the figure and plot the chart
airport_fig = go.Figure(data=[airport_trace], layout=airport_layout)
airport_fig.show()

## Average price by Continent 


import plotly.express as px

continent_avg = merged_df.groupby('Continent')['Price'].mean().sort_values(ascending=False)

colors = ['#FDEFEF', '#FAD2D2', '#F7B5B5', '#F49898', '#F17B7B']

fig = px.bar(x=continent_avg.index, y=continent_avg.values,
             color=continent_avg.values, color_continuous_scale=colors,
             labels={'x':'Continent', 'y':'Average Price'},
             title='Average Price by Continent')

fig.update_layout(title_x=0.5, plot_bgcolor='white',
                  xaxis=dict(showgrid=False, tickfont_size=14),
                  yaxis=dict(showgrid=False, tickfont_size=14,
                             title_font_size=16, tickprefix='$'))

fig.show()

#Beginner Slopes vs Price by Season

ski_resorts_df = merged_df[merged_df["Beginner slopes"] != "Unknown"]

fig1 = px.scatter(ski_resorts_df, x="Beginner slopes", y="Price", color="Season", size="Total lifts", hover_name="Resort", title="Beginner Slopes vs Price by Season")
fig1.update_layout(
    xaxis_title="Number of Beginner Slopes",
    yaxis_title="Price",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)'
)

##Total Slopes and Longest Run by Continent

fig2 = px.box(merged_df, x="Continent", y=["Total slopes", "Longest run"], title="Distribution of Total Slopes and Longest Run by Continent")
fig2.update_layout(
    yaxis_title="Number of Slopes or Length of Longest Run",
    yaxis_title_font=dict(
        family="Courier New, monospace",
        size=16,
        color="#7f7f7f"
    ),
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)'
)

## Map of Resorts and Airports 

import folium

map_resorts_airports = folium.Map(location=[48.690833, 9.140556], zoom_start=4)

for i, row in merged_df.iterrows():
    # add resort marker
    folium.Marker(
        location=[row['Latitude_x'], row['Longitude_x']],
        tooltip=row['Resort'],
        icon=folium.Icon(icon='circle', prefix='fa', color='blue', icon_color='white'),
        # set the size of the marker icon
        icon_size=(2, 2)
    ).add_to(map_resorts_airports)
    

    folium.Marker(
        location=[row['Latitude_y'], row['Longitude_y']],
        tooltip=row['Airport_Name'],
        icon=folium.Icon(icon='plane', prefix='fa', color='red'),
        # set the size of the marker icon
        icon_size=(2, 2)
    ).add_to(map_resorts_airports)
    
   
    folium.PolyLine(
        locations=[[row['Latitude_x'], row['Longitude_x']], [row['Latitude_y'], row['Longitude_y']]],
        color='black', dash_array='5',
    ).add_to(map_resorts_airports)

legend_html = """
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 120px; height: 90px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     ">&nbsp; Marker Legend <br>
     &nbsp; <i class="fa fa-circle fa-1x" style="color:blue"></i>&nbsp; Resort <br>
     &nbsp; <i class="fa fa-plane fa-1x" style="color:red"></i>&nbsp; Airport <br>
      </div>
     """
map_resorts_airports.get_root().html.add_child(folium.Element(legend_html))


title_html = '''
             <h3 align="center" style="font-size:20px"><b>Ski Resorts and Airports Map</b></h3>
             '''
map_resorts_airports.get_root().html.add_child(folium.Element(title_html))

map_resorts_airports


map_resorts_airports
map_resorts_airports.save('map_new.html')

## -------------------------------------Multiple Linear Regression Model------------------------------------ ##

from sklearn.linear_model import LinearRegression 
import statsmodels.api as sm

# create dummies for the "Continents" column
continent_dummies = pd.get_dummies(merged_df['Continent'])

# Join the dummies to the original DataFrame
merged_df = pd.concat([merged_df, continent_dummies], axis=1)

# Drop the original "Continents" column
merged_df = merged_df.drop('Continent', axis=1)

#Model 1
x = merged_df[['Oceania','North America','South America','Europe','Highest point','Lowest point',
'Beginner slopes', 'Intermediate slopes', 'Difficult slopes','Longest run', 'Snow cannons', 'Surface lifts',
'Chair lifts', 'Gondola lifts', 'Lift capacity',
'Child friendly', 'Snowparks', 'Nightskiing', 'Summer skiing','distance_miles']]
y = merged_df['Price']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=101)

lm = LinearRegression() 
lm.fit(x_train, y_train)

# Fit linear regression model
x_train_const = sm.add_constant(x_train)
lm = sm.OLS(y_train, x_train_const).fit()

# Print summary table
print(lm.summary())

#Model 2
country_dummies = pd.get_dummies(merged_df['Resort_Country'])
unique_countries = merged_df['Resort_Country'].unique()
print(unique_countries)
merged_df = pd.concat([merged_df, country_dummies], axis=1)

#Using Australia as base
x = merged_df[['Highest point','Lowest point',
               'Beginner slopes', 'Intermediate slopes', 'Difficult slopes','Longest run', 'Snow cannons', 'Surface lifts',
               'Chair lifts', 'Gondola lifts', 'Lift capacity','Child friendly', 'Snowparks', 'Nightskiing', 'Summer skiing','distance_miles',
               'Andorra', 'Argentina', 'Austria', 'Bosnia and Herzegovina', 'Bulgaria', 'Canada', 'Chile',
               'China', 'Czech Republic', 'Finland', 'France', 'Georgia', 'Germany', 'Greece', 'Iran', 'Italy', 'Japan',
               'Kazakhstan', 'Lebanon', 'Liechtenstein', 'Lithuania', 'New Zealand', 'Norway', 'Poland', 
               'Romania', 'Russia', 'Serbia', 'Slovakia', 'Slovenia', 'South Korea', 'Spain', 'Sweden',
               'Switzerland', 'Turkey', 'Ukraine', 'United Kingdom', 'United States']]
y = merged_df['Price']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=101)

from sklearn.linear_model import LinearRegression 
lm_2 = LinearRegression() 
lm_2.fit(x_train, y_train)

# Train linear regression model using statsmodels
x_train_const = sm.add_constant(x_train)
lm_2 = sm.OLS(y_train, x_train_const).fit()

# Display regression output
print(lm_2.summary())

## -------------------------------------Clustering------------------------------------ ##


merged_df1 = pd.merge(resorts, airports, left_on='nearest_index', right_index=True)
merged_df1[['Child friendly', 'Snowparks', 'Nightskiing', 'Summer skiing']] = merged_df1[['Child friendly', 'Snowparks', 'Nightskiing', 'Summer skiing']].replace({'Yes': 1, 'No': 0})



###--------------------------------Cluster Analysis - NA --------------------------------###
NA_subset = merged_df1[merged_df1['Continent'] == 'North America']

#Drop categorical variables

NA_subset.drop(['Resort', 'Resort_Country', 'Continent', 'Airport_City', 'Airport_Country',
                'Season', 'Airport_Name','Airport_ID', 'Latitude_x','Latitude_y','Longitude_x','Longitude_y',
                'nearest_index'], axis=1, inplace=True)

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

scaler=MinMaxScaler() #initialize
scaler.fit(NA_subset)
NA_scaled=scaler.transform(NA_subset)

wcv = []
silk_score = []
for i in range (2,20):
    
    km = KMeans(n_clusters = i, random_state = 0) 
    km.fit(NA_scaled) 

    wcv.append(km.inertia_) 
    silk_score.append(silhouette_score(NA_scaled, km.labels_))

plt.plot(range(2,20),wcv)
plt.xlabel('No. of Clusters')
plt.ylabel('Within Cluster Variation')
plt.xticks(range(2,20))

plt.grid()


km4_NA = KMeans(n_clusters = 4, random_state = 0)
km4_NA.fit(NA_scaled)
km4_NA.labels_

NA_subset ['labels'] = km4_NA.labels_

NA_subset.groupby('labels').mean()

#Interpretation: 
#Cluster 1
pd.set_option('display.max_columns', None)
Na_1 = NA_subset.loc[NA_subset['labels'] == 0].describe()

#Cluster 2

Na_2 = NA_subset.loc[NA_subset['labels'] == 1].describe()

#Cluster 3

Na_3 = NA_subset.loc[NA_subset['labels'] == 2].describe()

#Cluster 4

Na_4 = NA_subset.loc[NA_subset['labels'] == 3].describe()

NA_with_labels = pd.concat([NA_subset, merged_df], axis=1)


# select the rows where the labels column has a value of 1
mask = (NA_with_labels['labels'] == 1)

# select the resort names from the rows where the mask is True
NAresorts_with_label_1 = NA_with_labels.loc[mask, ['Resort','Resort_Country']]


# assume that the DataFrame containing resorts with label 1 is called `NAresorts_with_label_1`
# find the count of resorts in USA
usa_count = NAresorts_with_label_1[NAresorts_with_label_1['Resort_Country'] == 'United States']['Resort'].nunique()

# find the count of resorts in Canada
canada_count = NAresorts_with_label_1[NAresorts_with_label_1['Resort_Country'] == 'Canada']['Resort'].nunique()

print(f"Number of resorts in USA: {usa_count}")
print(f"Number of resorts in Canada: {canada_count}")

###--------------------------------Cluster Analysis - AS --------------------------------###
AS_subset = merged_df1[merged_df1['Continent'] == 'Asia']

#Drop categorical variables

AS_subset.drop(['Resort', 'Resort_Country', 'Continent', 'Airport_City', 'Airport_Country',
                'Season', 'Airport_Name','Airport_ID', 'Latitude_x','Latitude_y','Longitude_x','Longitude_y',
                'nearest_index'], axis=1, inplace=True)


scaler=MinMaxScaler() #initialize
scaler.fit(AS_subset)
AS_scaled=scaler.transform(AS_subset)

wcv = []
silk_score = []
for i in range (2,20):
    
    km = KMeans(n_clusters = i, random_state = 0) 
    km.fit(AS_scaled) 

    wcv.append(km.inertia_) 
   

plt.plot(range(2,20),wcv)
plt.xlabel('No. of Clusters')
plt.ylabel('Within Cluster Variation')
plt.xticks(range(2,20))

plt.grid()

km4_AS = KMeans(n_clusters = 4, random_state = 0)
km4_AS.fit(AS_scaled)
km4_AS.labels_

AS_subset ['labels'] = km4_AS.labels_

AS_subset.groupby('labels').mean()

#Interpretation: 
#Cluster 1
pd.set_option('display.max_columns', None)
As_1 = AS_subset.loc[AS_subset['labels'] == 0].describe()

#Cluster 2

As_2=AS_subset.loc[AS_subset['labels'] == 1].describe()

#Cluster 3

As_3=AS_subset.loc[AS_subset['labels'] == 2].describe()

#Cluster 4

As_4=AS_subset.loc[AS_subset['labels'] == 3].describe()


AS_with_labels = pd.concat([AS_subset, merged_df1], axis=1)


# select the rows where the labels column has a value of 1
mask_AS = (AS_with_labels['labels'] == 1)

# select the resort names from the rows where the mask is True
ASresorts_with_label_1 = AS_with_labels.loc[mask_AS, ['Resort','Resort_Country']]


###--------------------------------Cluster Analysis - EU --------------------------------###

EU_subset = merged_df1[merged_df1['Continent'] == 'Europe']

#Drop categorical variables

EU_subset.drop(['Resort', 'Resort_Country', 'Continent', 'Airport_City','Airport_Country','Season', 'Airport_Name','Airport_ID', 'Latitude_x','Latitude_y','Longitude_x','Longitude_y',
                'nearest_index'], axis=1, inplace=True)

scaler=MinMaxScaler() #initialize
scaler.fit(EU_subset)
EU_scaled=scaler.transform(EU_subset)

wcv = []
silk_score = []
for i in range (2,25):
    
    km = KMeans(n_clusters = i, random_state = 0) 
    km.fit(EU_scaled) 

    wcv.append(km.inertia_)
   

plt.plot(range(2,25),wcv)
plt.xlabel('No. of Clusters')
plt.ylabel('Within Cluster Variation')
plt.xticks(range(2,25))

plt.grid()

km4_EU = KMeans(n_clusters = 4, random_state = 0)
km4_EU.fit(EU_scaled)
km4_EU.labels_

EU_subset['labels'] = km4_EU.labels_

EU_subset.groupby('labels').mean()

#Interpretation: 
#Cluster 1
pd.set_option('display.max_columns', None)
Eu_1 = EU_subset.loc[EU_subset['labels'] == 0].describe()

#Cluster 2

Eu_2 = EU_subset.loc[EU_subset['labels'] == 1].describe()

#Cluster 3

Eu_3 = EU_subset.loc[EU_subset['labels'] == 2].describe()

#Cluster 4

Eu_4 = EU_subset.loc[EU_subset['labels'] == 3].describe()


EU_with_labels = pd.concat([EU_subset, merged_df1], axis=1)


# select the rows where the labels column has a value of 1
mask_EU = (EU_with_labels['labels'] == 1)

# select the resort names from the rows where the mask is True
EUresorts_with_label_1 = EU_with_labels.loc[mask_EU, ['Resort','Resort_Country']]

country_counts = EUresorts_with_label_1['Resort_Country'].value_counts()

print(country_counts)

###--------------------------------Cluster Analysis - OC --------------------------------###

OC_subset = merged_df1[merged_df1['Continent'] == 'Oceania']

#Drop categorical variables

OC_subset.drop(['Resort', 'Resort_Country', 'Continent', 'Airport_City','Airport_Country','Season', 'Airport_Name','Airport_ID', 'Latitude_x','Latitude_y','Longitude_x','Longitude_y',
                'nearest_index'], axis=1, inplace=True)

scaler=MinMaxScaler() #initialize
scaler.fit(OC_subset)
OC_scaled=scaler.transform(OC_subset)

wcv = []
silk_score = []
for i in range (2,20):
    
    km = KMeans(n_clusters = i, random_state = 0) 
    km.fit(EU_scaled) 

    wcv.append(km.inertia_)
   

plt.plot(range(2,20),wcv)
plt.xlabel('No. of Clusters')
plt.ylabel('Within Cluster Variation')
plt.xticks(range(2,20))
plt.grid()

km4_OC = KMeans(n_clusters = 4, random_state = 0)
km4_OC.fit(OC_scaled)
km4_OC.labels_

OC_subset['labels'] = km4_OC.labels_

OC_subset.groupby('labels').mean()

#Interpretation: 
#Cluster 1
pd.set_option('display.max_columns', None)
Oc_1 = OC_subset.loc[OC_subset['labels'] == 0].describe()

#Cluster 2

Oc_2 = OC_subset.loc[OC_subset['labels'] == 1].describe()
#Cluster 3

Oc_3 = OC_subset.loc[OC_subset['labels'] == 2].describe()
#Cluster 4

Oc_4 = OC_subset.loc[OC_subset['labels'] == 3].describe()


OC_with_labels = pd.concat([OC_subset, merged_df1], axis=1)


# select the rows where the labels column has a value of 1
mask_OC = (OC_with_labels['labels'] == 1)

# select the resort names from the rows where the mask is True
OCresorts_with_label_1 = OC_with_labels.loc[mask_OC, ['Resort','Resort_Country']]

###--------------------------------Cluster Analysis - SA --------------------------------###

SA_subset = merged_df1[merged_df1['Continent'] == 'South America']

#Drop categorical variables

SA_subset.drop(['Resort', 'Resort_Country', 'Continent', 'Airport_City','Airport_Country','Season', 'Airport_Name','Airport_ID', 'Latitude_x','Latitude_y','Longitude_x','Longitude_y',
                'nearest_index'], axis=1, inplace=True)

scaler=MinMaxScaler() #initialize
scaler.fit(SA_subset)
SA_scaled=scaler.transform(SA_subset)

n_clusters = range(2,19)
wcv = []

for i in n_clusters:
    km = KMeans(n_clusters=i, random_state=0) 
    km.fit(EU_scaled)
    wcv.append(km.inertia_)

plt.plot(n_clusters, wcv)
plt.xlabel('No. of Clusters')
plt.ylabel('Within Cluster Variation')
plt.xticks(n_clusters)
plt.grid()

      
km4_SA = KMeans(n_clusters = 4, random_state = 0)
km4_SA.fit(SA_scaled)
km4_SA.labels_


SA_subset['labels'] = km4_SA.labels_


SA_subset.groupby('labels').mean()

#Interpretation: 
#Cluster 1
pd.set_option('display.max_columns', None)
Sa_1 = SA_subset.loc[SA_subset['labels'] == 0].describe()

#Cluster 2

Sa_2 = SA_subset.loc[SA_subset['labels'] == 1].describe()
#Cluster 3

Sa_3 = SA_subset.loc[SA_subset['labels'] == 2].describe()
#Cluster 4

Sa_4 = SA_subset.loc[SA_subset['labels'] == 3].describe()


SA_with_labels = pd.concat([SA_subset, merged_df1], axis=1)


# select the rows where the labels column has a value of 1
mask_SA = (SA_with_labels['labels'] == 2)

# select the resort names from the rows where the mask is True
SAresorts_with_label_1 = SA_with_labels.loc[mask_SA, ['Resort','Resort_Country']]

