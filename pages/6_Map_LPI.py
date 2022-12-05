########################################
# Import libraries
########################################
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

########################################
# Configure Page
########################################
st.set_page_config(page_title="Test", page_icon="🥑",layout="wide")

########################################
# Add Title
########################################
st.markdown("# LPI")
st.sidebar.header("LPI - Choropleth Map")

########################################
# Read Data
########################################
df = pd.read_csv("data\LPI_Data.csv", encoding='latin-1')

########################################
# Create Filters
########################################
years = st.select_slider(
    'Select years', 
    options=['2007', '2010', '2012', '2014', '2016', '2018'],
    value=('2007', '2018')
    )

series = st.radio(
    "Select rank or score.",
    ('rank', 'score'), horizontal= True)

# Create List for series_name selectbox
series_list=df[df['Series Name'].str.contains(series,na=False)]
series_list=series_list['Series Name'].unique()

series_name = st.selectbox(
        "Choose overall score or category", list(series_list)
    )

########################################
# Clean Data 
# Change Data Types
# Filter for rank or score
########################################
df=df.replace(to_replace='..', value=None)
df_series=df[df['Series Name'].str.contains(series,na=False)]
df_series=df_series[df_series['Series Name']==series_name]
convert_dtypes_dict={
    'Country Name':object,
    'Country Code':object,
    'Series Name':object,
    'Series Code':object,
    '2007 [YR2007]':float,
    '2010 [YR2010]':float,
    '2012 [YR2012]':float,
    '2014 [YR2014]':float,
    '2016 [YR2016]':float,
    '2018 [YR2018]':float
    }
df_series = df_series.astype(convert_dtypes_dict)

########################################
# Filter Dataset
# Build Mean for selected timeframe
########################################
year_dict= {
    '2007':'2007 [YR2007]',
    '2010':'2010 [YR2010]',
    '2012':'2012 [YR2012]',
    '2014':'2014 [YR2014]',
    '2016':'2016 [YR2016]',
    '2018':'2018 [YR2018]'
}

df_series['mean']=df_series.loc[:,year_dict[years[0]]:year_dict[years[1]]].mean(axis=1)

########################################
# Prepare Data for Visualisation
########################################
df_map=df_series[['Country Name','mean']]

########################################
# Create Visualisation
########################################
fig = go.Figure(data=go.Choropleth(
    locations=df_map['Country Name'], # Spatial coordinates
    z = df_map['mean'].astype(float), # Data to be color-coded
    locationmode = 'country names', # set of locations match entries in `locations`
    colorscale = 'Greens',
    colorbar_title = series,
))

fig.update_layout(
    title_text = series_name,
    title_x=0.5,
    geo_scope='world', 
    template='simple_white',
    width=1500, height=1000

)
# fig.update_traces(
#     hovertemplate = "%{locations}<br>%{z}%"
# )

########################################
# Show Visualisation
########################################
st.plotly_chart(fig, use_container_width=True)